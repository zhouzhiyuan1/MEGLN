import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
import json
import argparse
import pandas as pd
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import datetime
from im2mesh.checkpoints import CheckpointIO
import os
from im2mesh import config
import time

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=6, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

stats_dict = {}

kwargs = {}

# Preprocess if requires

eval_dicts = []

eval_dict = {}

# Compute elbo




it = -1
# Compute iou






for batch in test_loader:
    it += 1
    now = time.time()

    points = batch.get('points').to(device)
    occ = batch.get('points.occ').to(device)

    inputs = batch.get('inputs', torch.empty(points.size(0), 0)).to(device)
    voxels_occ = batch.get('voxels')

    points_iou = batch.get('points_iou').to(device)
    occ_iou = batch.get('points_iou.occ').to(device)
    with torch.no_grad():
        p_out0, p_out1, p_out2, p_out3, p_out4 = model(points_iou, inputs, sample=False, **kwargs)
    occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
    occ_iou_hat_np = (p_out4.probs >= 0.5).cpu().numpy()
    fs = classification_report(occ_iou_np, occ_iou_hat_np, digits=3, output_dict=True)
    #print('samplesAvg1', fs['samples avg'])
    # eval_dict['microAvg'] = fs['micro avg']
    # eval_dict['macroAvg'] = fs['macro avg']
    # eval_dict['weightedAvg'] = fs['weighted avg']
    # eval_dict['samplesAvg'] = fs['samples avg']
    #print('samplesAvg2', eval_dict['samplesAvg'])
    eval_dicts.append(fs['micro avg'])
    print(
        'it=%03d,per_iter_time=%.3f'
        % (it, time.time() - now))


out_file_class = os.path.join('/data1/lab105/zhouzhiyuan/MEGLN', 'eval_fscore_micro.csv')
eval_df = pd.DataFrame(eval_dicts)
# Create CSV file
eval_df.to_csv(out_file_class)



