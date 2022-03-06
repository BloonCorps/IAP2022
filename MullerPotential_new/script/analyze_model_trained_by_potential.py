__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 21:50:25"

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import torch.distributions as distributions
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from RealNVP import *
from functions import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

data = torch.load("./output/model_trained_by_potential/model_alpha_{:.3f}_step_{}.pt".format(alpha, 24))
hidden_dim = data['hidden_dim']
masks = data['masks']
x1_min, x1_max = data['masks'][-2][2], data['masks'][-2][3]
x2_min, x2_max = data['masks'][-1][2], data['masks'][-1][3]

realNVP = RealNVP(masks, hidden_dim)

num_steps = 1000
x_from_z_plots = PdfPages(f"./output/analyze_model_trained_by_potential/x_transformed_from_z_learned_by_potential_alpha_{alpha:.3f}.pdf")
z_from_x_plots = PdfPages(f"./output/analyze_model_trained_by_potential/z_transformed_from_x_learned_by_potential_alpha_{alpha:.3f}.pdf")

DKL = []
idx_step_list = []

for idx_step in range(num_steps):
    if (idx_step + 1) % 25 == 0:
        print("idx_step: {}".format(idx_step))
        
        data = torch.load("./output/model_trained_by_potential/model_alpha_{:.3f}_step_{}.pt".format(alpha, idx_step))
        realNVP.load_state_dict(data['state_dict'])
        
        z = torch.normal(0, 1, size = (10000, 2), device = next(realNVP.parameters()).device)
        with torch.no_grad():
            x, logdet = realNVP(z)
        x = x.cpu().detach().numpy()
        z = z.cpu().detach().numpy()

        ## split data points into two sets based on x     
        flag = x[:,1] > x[:,0] + 1.5
        fig = plt.figure(0)
        fig.clf()
        plt.plot(x[flag,0], x[flag,1], ".", alpha = 0.2)
        plt.plot(x[~flag,0], x[~flag,1], ".", alpha = 0.2)
        plt.xlabel(r'$X_1$')
        plt.ylabel(r'$X_2$')
        plt.xlim([x1_min, x1_max])
        plt.ylim([x2_min, x2_max])
        x_from_z_plots.savefig()

        fig = plt.figure(0)
        fig.clf()
        plt.plot(z[flag,0], z[flag,1], ".", alpha = 0.2)
        plt.plot(z[~flag,0], z[~flag,1], ".", alpha = 0.2)
        plt.xlabel(r'$z_1$')
        plt.ylabel(r'$z_2$')
        z_from_x_plots.savefig()

        DKL.append([])
        idx_step_list.append(idx_step)
        
        for r in range(10):
            normal_dist = distributions.Normal(0.0, 1.0)
            z = normal_dist.sample((10000, 2))
            z.to(next(realNVP.parameters()).device)
            with torch.no_grad():
                x, logdet = realNVP(z)

            energy = compute_Muller_potential(alpha, x)
            DKL_samples = torch.sum(normal_dist.log_prob(z), -1) - logdet + energy
            DKL_value = torch.mean(DKL_samples).item()
            DKL[-1].append(DKL_value)
        
x_from_z_plots.close()
z_from_x_plots.close()

DKL = np.array(DKL)
DKL_mean = np.mean(DKL, -1)
DKL_std = np.std(DKL, -1)
with open(f"./output/analyze_model_trained_by_potential/DKL_alpha_{alpha:.3f}.pkl", 'wb') as file_handle:
    pickle.dump({'DKL':DKL, 'idx_step_list': idx_step_list}, file_handle)        

fig = plt.figure(0)
fig.clf()
plt.errorbar(idx_step_list, DKL_mean, yerr = DKL_std)
plt.savefig(f"./output/analyze_model_trained_by_potential/DKL_alpha_{alpha:.3f}.eps")
