__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/12/14 17:20:28"

import numpy as np
import torch
import pickle
import scipy.integrate as integrate
from functions import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open("./output/range.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    x1_min, x1_max = data['x1_min'], data['x1_max']
    x2_min, x2_max = data['x2_min'], data['x2_max']    

def compute_Muller_prob(x2, x1, alpha):
    x = torch.tensor([[x1, x2]])
    U = compute_Muller_potential(alpha, x)
    U = U.item()    
    return np.exp(-U)

Z, Z_error = integrate.dblquad(compute_Muller_prob, x1_min, x1_max, x2_min, x2_max, [alpha])
F = -np.log(Z)
print("alpha = {:.3f}, F using numerical integration: {:.3f}".format(alpha, F))

with open("./output/F_numeric_integration_alpha_{:.3f}.pkl".format(alpha), 'wb') as file_handle:
    pickle.dump({'Z': Z, 'Z_error': Z_error, 'F': F}, file_handle)
