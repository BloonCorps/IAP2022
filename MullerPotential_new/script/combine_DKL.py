__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/12/21 22:05:34"

import numpy as np
import torch
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open(f"./output/analyze_model_trained_by_data/DKL_alpha_{alpha:.3f}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    DKL_by_data = data['DKL']
    idx_step_list_by_data = data['idx_step_list']

with open(f"./output/analyze_model_trained_by_potential/DKL_alpha_{alpha:.3f}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    DKL_by_potential = data['DKL']
    idx_step_list_by_potential = data['idx_step_list']
    
with open("./output/F_numeric_integration_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    Z = data['Z']
    Z_error = data['Z_error']
    F = data['F']

    
fig = plt.figure(0)
fig.clf()
plt.errorbar(idx_step_list_by_data, -np.mean(DKL_by_data, -1), yerr = np.std(DKL_by_data, -1), label = 'DKL_by_data')
plt.errorbar(idx_step_list_by_potential, np.mean(DKL_by_potential, -1), yerr = np.std(DKL_by_potential, -1), label = "DKL_by_potential")
plt.hlines(F, xmin = 0, xmax = np.max(idx_step_list_by_data), label = "F_by_numerical_integration")
plt.legend()
plt.savefig("./output/combined_DKL.pdf")



