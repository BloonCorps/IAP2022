__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/12/14 16:07:12"

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from FastMBAR import *
from functions import *
from sys import exit
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    x_record = data['x_record']
    alphas = data['alphas']

x_record = np.transpose(x_record, (1,0,2))
x_record = x_record.reshape((-1, 2))
x_record = torch.from_numpy(x_record)
energy = compute_Muller_potential(1.0, x_record)
energy = alphas.reshape((-1, 1)) * energy
num_conf = np.array([energy.shape[-1]//energy.shape[0] for i in range(len(alphas))])
fastmbar = FastMBAR(energy.numpy(), num_conf, cuda = True, verbose = True)

F = fastmbar.F

data = torch.load("./output/model_trained_by_potential/model_alpha_{:.3f}.pt".format(alpha))
masks = data['masks']
x1_min, x1_max = data['masks'][-2][2], data['masks'][-2][3]
x2_min, x2_max = data['masks'][-1][2], data['masks'][-1][3]
F = F - np.log((x1_max - x1_min)*(x2_max - x2_min))

with open("./output/TREMC/free_energy_alpha_{:.3f}.pkl".format(alpha), 'wb') as file_handle:
    pickle.dump({'F': F, 'alphas': alphas}, file_handle)

