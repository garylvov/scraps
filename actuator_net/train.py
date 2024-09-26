from utils import train_actuator_network_and_plot_predictions
from glob import glob
import datetime

# Evaluates the existing actuator network by default
# load_pretrained_model = True
# actuator_network_path = "../../resources/actuator_nets/unitree_go1.pt"

experiment_dir = '/home/garylvov/hashi_ws/src/data'
now = datetime.datetime.now()
model_type = 'lstm'
actuator_network_path = f"hashi.pt"
dataloader_path = f"hashi.dataloader"
# Uncomment these lines to train a new actuator network
load_pretrained_model = False
train_actuator_network_and_plot_predictions(experiment_dir, 
                                            actuator_network_path=actuator_network_path, 
                                            dataloader_path=dataloader_path,
                                            model_type=model_type,
                                            load_pretrained_model=load_pretrained_model)
