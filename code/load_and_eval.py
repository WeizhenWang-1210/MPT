import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cuda, normalize
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
from prerender.utils.utils import data_to_numpy, get_config
import subprocess
from matplotlib import pyplot as plt
import os
import glob
import sys
import random
from my_utils import Metric, Visualizer

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_last_file(path):
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

config = get_config(sys.argv[1])
# alias = 'run' # sys.argv[1].split("/")[-1].split(".")[0][:3]
mask = config['train']['data_config']['dataset_config']['mask_history_fraction']
if not os.path.exists(f"mask={mask}"):
    os.mkdir(f"mask={mask}")
if not os.path.exists(f"mask={mask}/models"):
    os.mkdir(f"mask={mask}/models")
models_path = f"mask={mask}/models/run__72a81b0"
if not os.path.exists(models_path):
    os.mkdir(models_path)
if not os.path.exists(f"mask={mask}/metrics_transformer_redo"):
    os.makedirs(f"mask={mask}/metrics_transformer_redo")
    
if __name__ == "__main__":
    
    for step in range(1368, 24625, 1368):
        last_checkpoint = f"{models_path}/{step}.pth"
        val_dataloader = get_dataloader(config["val"]["data_config"])
        model = MultiPathPP(config["model"])
        model.cuda()
        model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])
        model.eval()



        with torch.no_grad():
            torch.cuda.empty_cache()
            minADEs = []
            minFDEs = []
            missRates = []
            
            for b, data in enumerate(tqdm(val_dataloader)):
                if config["train"]["normalize"]:
                    data = normalize(data, config)
                dict_to_cuda(data)
                probas, coordinates, _, _ = model(data, step)
                if config["train"]["normalize_output"]:
                    coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
                    
                fut_valid = torch.all(torch.squeeze(data["target/future/valid"]), dim=1)
                his_valid = torch.all(torch.squeeze(data["target/history/valid"]), dim=1).cuda()
                valid = torch.logical_and(fut_valid, his_valid)
                if torch.any(valid):
                    metric = Metric(coordinates[valid], data["target/future/xy"][valid], probas[valid])
                    minADEs.append(metric.minADE().detach().cpu().numpy())
                    minFDEs.append(metric.minFDE().detach().cpu().numpy())
                    missRates.append(metric.allMissRates())

            with open(f'mask={mask}/metrics_transformer_redo/step={step}.txt', 'w') as f:
                f.writelines("\t".join([
                    str(np.mean(np.array(minADEs))),
                    str(np.mean(np.array(minFDEs))),
                    str(np.mean(np.array(missRates)))
                ]))