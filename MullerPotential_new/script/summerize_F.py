__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/12/14 18:12:32"

import numpy as np
import torch
import pickle
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

## results from numeric integration
with open("./output/F_numeric_integration_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data =  pickle.load(file_handle)
    Z = data['Z']
    Z_error = data['Z_error']
    F = data['F']
print("F from numerical integration: {:.3f}".format(F))

with open("./output/TREMC/free_energy_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data =  pickle.load(file_handle)
    F = data['F']
print("F from MC: {:.3f}".format(F[-1]))

with open("./output/analyze_model_trained_by_potential/DKL_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    DKL =  pickle.load(file_handle)
    DKL_mean = np.mean(DKL)
print("DKL (trained by potential): {:.3f}".format(DKL_mean))

with open("./output/analyze_model_trained_by_data/DKL_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    DKL =  pickle.load(file_handle)
    DKL_mean = np.mean(DKL)
print("DKL (trained by data): {:.3f}".format(DKL_mean))

