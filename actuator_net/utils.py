# Modified from:
# https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/actuator_net/utils.py
# By Gary Lvov
import os
import pickle as pkl
from matplotlib import pyplot as plt
import time
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
import datetime

class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['joint_states'])

    def __getitem__(self, idx):
        return {k: v[idx] for k,v in self.data.items()}

class Act(nn.Module):
    def __init__(self, act, slope=0.05):
        super(Act, self).__init__()
        self.act = act
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input):
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1.)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1.) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1.)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1.) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1.) - self.shift
        elif self.act == "leaky_ssp":
            return (
                F.softplus(input, beta=1.) -
                self.slope * F.relu(-input) -
                self.shift
            )
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        elif self.act == "softsign":
            return F.softsign(input)
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")

def build_mlp(in_dim, units, layers, out_dim, act='relu', layer_norm=False, act_final=False):
    mods = [nn.Linear(in_dim, units), Act(act)]
    for i in range(layers-1):
        mods += [nn.Linear(units, units), Act(act)]
    mods += [nn.Linear(units, out_dim)]
    if act_final:
        mods += [Act(act)]
    if layer_norm:
        mods += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*mods)

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, out_dim):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def build_lstm(in_dim, units, layers, out_dim):
    return LSTMModel(in_dim, units, layers, out_dim)

def save_dataloaders(train_loader, test_loader, save_path):
    dataloaders = {'train': train_loader, 'test': test_loader}
    with open(save_path, 'wb') as f:
        pkl.dump(dataloaders, f)

def load_dataloaders(load_path):
    with open(load_path, 'rb') as f:
        dataloaders = pkl.load(f)
    return dataloaders['train'], dataloaders['test']

def train_actuator_network(xs, ys, batch_size, num_samples_in_history, units, layers, lr, epochs, eps, weight_decay,
                           actuator_network_path, dataloader_path, model_type, save_dataloaders_flag=True,
                           return_stats=False):
    print(xs.shape, ys.shape)
    num_data = xs.shape[0]
    num_train = num_data // 5 * 4
    num_test = num_data - num_train

    dataset = ActuatorDataset({"joint_states": xs, "tau_ests": ys})
    train_set, val_set = random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if save_dataloaders_flag:
        save_dataloaders(train_loader, test_loader, dataloader_path)
    
    if model_type == "mlp":
        model = build_mlp(in_dim=(num_samples_in_history + 1) * 2, 
                        units=units, layers=layers, out_dim=1, act='softsign')
    elif model_type == "lstm":
        model = build_lstm(1, units=units, layers=layers, out_dim=1)

    opt = Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    device = 'cuda:0'
    model = model.to(device)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M%p")
    run_name = f"bs{batch_size}_u{units}_l{layers}_lr{lr}_eps{eps}_wd{weight_decay}_ns{num_samples_in_history}_{current_time}"
    log_dir = f'./logs/{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Hyperparameters dict for TensorBoard
    hparams = {
        'lr': lr,
        'batch_size': batch_size,
        'units': units,
        'layers': layers,
        'eps': eps,
        'weight_decay': weight_decay,
        'num_samples_in_history': num_samples_in_history
    }

    # Empty dict for any metrics you might want to track along with hyperparameters
    metrics = {
        'test_loss': 0,
        'mae': 0
    }

    # Start with logging the hyperparameters and initial metrics
    writer.add_hparams(hparams, metrics)

    for epoch in range(epochs):
        epoch_loss = 0
        ct = 0
        for batch in train_loader:
            data = batch['joint_states'].to(device)
            if model_type == 'lstm':
                data = data.view(data.size(0), data.size(1), 1) 
            y_pred = model(data)
            opt.zero_grad()
            y_label = batch['tau_ests'].to(device)
            loss = ((y_pred - y_label) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().cpu().numpy()
            ct += 1
        epoch_loss /= ct

        test_loss = 0
        mae = 0
        ct = 0
        with torch.no_grad():
            for batch in test_loader:
                data = batch['joint_states'].to(device)
                if model_type == 'lstm':
                    data = data.view(data.size(0), data.size(1), 1) 
                y_pred = model(data)
                y_label = batch['tau_ests'].to(device)
                tau_est_loss = ((y_pred - y_label) ** 2).mean()
                loss = tau_est_loss
                test_mae = (y_pred - y_label).abs().mean()

                test_loss += loss
                mae += test_mae
                ct += 1
            test_loss /= ct
            mae /= ct

        # Log losses and MAE for each epoch within the same hparam context
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('MAE/test', mae, epoch)

        print(f'epoch: {epoch} | loss: {epoch_loss:.4f} | test loss: {test_loss:.4f} | mae: {mae:.4f}')

        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(actuator_network_path)  # Save
        
    metrics['test_loss'] = test_loss
    metrics['mae'] = mae
    writer.add_hparams(hparams, metrics)
    writer.close()
    
    if return_stats:
        return model, test_loss.to('cpu').numpy(), mae.to('cpu').numpy()
    else:
        return model

def load_experiments(exp_dir, 
                    torque_scaling=.01,
                    torque_cliping=[-2, 2]):
    datas = []

    experiments = glob(f"{exp_dir}/*.pkl")
    for experiment in experiments:
        with open(experiment, 'rb') as f:
            data = pickle.load(f)
            # Maybe good to delineate trials somehow like this: (or better to just run all trials at once)
            # data.extend({
            # "joint_names": ["platform", "pitch", "roll"],
            # "joint_positions": [np.nan, np.nan, np.nan],
            # "joint_velocities": [np.nan, np.nan, np.nan],
            # "joint_efforts": [np.nan, np.nan, np.nan],
            # "time_sec": np.nan,
            # "time_nsec": np.nan,
            # "joint_position_command": [np.nan, np.nan, np.nan],
            #  })
            datas.extend(data) # shamelessly combine trials (messes with transitions
                               # between experiments, but messed up transitions have far less samples than other transitions
                               # so it should be fine. )

    # TODO: for now, we ignore the platform joint, needs to be added back eventually
    num_actuators = len(datas[0]["joint_positions"]) - 1

    tau_ests = np.zeros((len(datas), num_actuators))
    joint_positions = np.zeros((len(datas), num_actuators))
    joint_position_targets = np.zeros((len(datas), num_actuators))
    joint_velocities = np.zeros((len(datas), num_actuators))

    for i in range(len(datas)):
        # TODO: For now, we ignore platform joint. Needs to be added back
        tau_ests[i, :] = np.clip(np.array(datas[i]["joint_efforts"][1:]) * torque_scaling, 
                                 *torque_cliping)
        joint_positions[i, :] = datas[i]["joint_positions"][1:]
        joint_position_targets[i, :] = datas[i]["joint_position_command"][1:]
        joint_velocities[i, :] = datas[i]["joint_velocities"][1:]

    joint_position_errors = joint_positions - joint_position_targets
    joint_velocities = joint_velocities

    joint_position_errors = torch.tensor(joint_position_errors, dtype=torch.float)
    joint_velocities = torch.tensor(joint_velocities, dtype=torch.float)
    tau_ests = torch.tensor(tau_ests, dtype=torch.float)

    return joint_position_errors, joint_velocities, tau_ests, num_actuators

def prepare_data_for_model(joint_position_errors, joint_velocities, tau_ests, num_actuators, num_samples_in_history):
    xs = []
    ys = []
    
    num_samples_in_history += 1 # Include current time step
    # Loop over each actuator
    for i in range(num_actuators):
        # Create list to hold time-shifted features for current actuator
        xs_joint = []

        # Append time-shifted data for each time step from num_samples_in_history-1 to 0
        for t in range(num_samples_in_history-1, -1, -1):
            xs_joint.append(joint_position_errors[t:-(num_samples_in_history-t) if num_samples_in_history-t != 0 else None, i:i+1])
            xs_joint.append(joint_velocities[t:-(num_samples_in_history-t) if num_samples_in_history-t != 0 else None, i:i+1])

        # Concatenate all features horizontally (new feature columns)
        xs_joint = torch.cat(xs_joint, dim=1)
        xs.append(xs_joint)

        # The corresponding target (tau_ests) should be aligned with the last feature time step
        tau_ests_joint = tau_ests[num_samples_in_history-1:, i:i+1]
        ys.append(tau_ests_joint)

    # Concatenate all data vertically (stacking different actuators)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return xs[::num_samples_in_history + 1], ys[::num_samples_in_history + 1]

def train_actuator_network_and_plot_predictions(experiment_dir, actuator_network_path, dataloader_path, model_type, load_pretrained_model=False):
    hyperparam_sweep = True
    joint_position_errors, joint_velocities, tau_ests, num_actuators = load_experiments(experiment_dir,
                                                                                                                              torque_scaling=.01,
                                                                                                                              torque_cliping=[-2, 2])
    num_samples_in_history = 2
    train_xs, train_ys = prepare_data_for_model(joint_position_errors, joint_velocities, tau_ests, num_actuators, num_samples_in_history)

    if load_pretrained_model:
        model = torch.jit.load(actuator_network_path).to('cpu')
        train_loader, test_loader = load_dataloaders(dataloader_path) # test still test
    else:
        if hyperparam_sweep:
            param_grid = {
                'batch_size': [64],
                'units': [32, 48, 64],#32, 48
                'layers': [2, 3, 4],
                'lr': [8e-4, 8e-3, 1e-4],
                'eps': [1e-8],  # Values for the optimizer's epsilon
                'weight_decay': [0.0,   1e-8],  # Regularization term
                'num_samples_in_history': [2, 3, 4, 5, 6],  # Number of past samples to consider
                'epochs': [200]  # Reduced for quick experiments
            }
            # param_grid = {
            #     'batch_size': [128],
            #     'units': [32],#32, 48
            #     'layers': [2, 3,],
            #     'lr': [8e-4],
            #     'eps': [1e-8],  # Values for the optimizer's epsilon
            #     'weight_decay': [0.0],  # Regularization term
            #     'num_samples_in_history': [2],  # Number of past samples to consider
            #     'epochs': [100]  # Reduced for quick experiments
            # }

            grid = ParameterGrid(param_grid)
            results = []
            for params in tqdm(grid):
                try:
                    print(f"Attempting to train with hyperparameters: {params}")
                    #Configure and train the model with current set of hyperparameters
                    train_xs, train_ys = prepare_data_for_model(joint_position_errors, joint_velocities, tau_ests, num_actuators, 
                                                                params['num_samples_in_history'])
                    (model, test_loss, test_mae) = train_actuator_network(train_xs, train_ys,
                                                batch_size=params['batch_size'],
                                                num_samples_in_history=params['num_samples_in_history'],
                                                units=params['units'],
                                                layers=params['layers'],
                                                lr=params['lr'],
                                                epochs=params['epochs'],
                                                eps=params['eps'],
                                                weight_decay=params['weight_decay'],
                                                actuator_network_path=actuator_network_path,
                                                dataloader_path=dataloader_path,
                                                model_type=model_type,
                                                return_stats=True)
                    
                    #Load the test set and evaluate the model
                    _, test_loader = load_dataloaders(dataloader_path)
                    
                    # Append results
                    results.append((params, (float(test_loss), float(test_mae))))
                except Exception as e:
                    results.append((params, (float('inf'), float('inf'))))
                    print(f"Failed with hyperparameters: {params}")
                    print(e)
                    continue
            print(results)
            np.save("results.npy", np.array(results))
            best_params = min(results, key=lambda x: x[1][0])  # assuming test_loss is the first element in the tuple
            print(f"Best params based on test loss: {best_params[0]} with loss: {best_params[1][0]} and MAE: {best_params[1][1]}")
        else:
            model = train_actuator_network(train_xs, train_ys, batch_size=64,
                                        num_samples_in_history=num_samples_in_history, 
                                        units=32, layers=3, lr=8e-4, epochs=200, eps=1e-8, 
                                        weight_decay=0.0,
                                        actuator_network_path=actuator_network_path, 
                                        dataloader_path=dataloader_path,
                                        model_type=model_type).to("cpu")
            
            train_loader, test_loader = load_dataloaders(dataloader_path) # test still test after running eval.py
    
    if not hyperparam_sweep:
        # Predict and plot only for validation set
        val_xs = []
        val_ys = []
        
        num_samples = 200  # Change this to the desired number of samples
        for batch in test_loader:
            data = batch['joint_states']
            target = batch['tau_ests']
            val_xs.append(data)
            val_ys.append(target)
            if len(val_xs) >= num_samples:
                break
        val_xs = torch.cat(val_xs)[:num_samples]
        val_ys = torch.cat(val_ys)[:num_samples]
        if model_type == 'lstm':
            val_xs = val_xs.view(val_xs.size(0), val_xs.size(1), 1) 
        tau_preds = model(val_xs).detach().reshape(num_actuators, -1).T

        timesteps = np.linspace(0, 1, int(num_samples/2))
        tau_ests = val_ys.reshape(num_actuators, -1).T
    
        fig, axs = plt.subplots(1, num_actuators, figsize=(14, 6))
        axs = np.array(axs).flatten()
        print(timesteps.shape)
        print(tau_preds.shape)
        print(tau_ests.shape)
        for i in range(num_actuators):
            axs[i].plot(timesteps, tau_ests[:, i], label="Measured Torque (Y)", color="green", linewidth=.5)
            axs[i].plot(timesteps, tau_preds[:, i], label="Predicted Torque (Y_hat)", color="red", linewidth=.5)
            axs[i].legend()

        plt.show()