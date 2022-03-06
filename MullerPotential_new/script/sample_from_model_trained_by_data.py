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

data = torch.load("./output/model_trained_by_data/model_alpha_{:.3f}_step_24.pt".format(alpha))
hidden_dim = data['hidden_dim']
masks = data['masks']
x1_min, x1_max = data['masks'][-2][2], data['masks'][-2][3]
x2_min, x2_max = data['masks'][-1][2], data['masks'][-1][3]

with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    x_record = data['x_record']
    alphas = data['alphas']    
x_train = x_record[:, -1, :]
realNVP = RealNVP(masks, hidden_dim)
x_train = torch.from_numpy(x_train)
x_train = x_train.to(next(realNVP.parameters()).device)

num_steps = 3000
x_from_z_plots = PdfPages(f"./output/analyze_model_trained_by_data/x_transformed_from_z_alpha_{alpha:.3f}.pdf")
z_from_x_plots = PdfPages(f"./output/analyze_model_trained_by_data/z_transformed_from_x_alpha_{alpha:.3f}.pdf")

for idx_step in range(num_steps):
    if (idx_step + 1) % 25 == 0:
        print("idx_step: {}".format(idx_step))
        data = torch.load("./output/model_trained_by_data/model_alpha_{:.3f}_step_{}.pt".format(alpha, idx_step))
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
        plt.title("idx_step: {}".format(idx_step))
        x_from_z_plots.savefig()

        ## split data points into two sets based on x     
        flag = x_train[:,1] > x_train[:,0] + 1.5 
        with torch.no_grad():
            z, _ = realNVP.inverse(x_train)
        z = z.cpu().detach().numpy()
        
        fig = plt.figure(0)
        fig.clf()
        plt.plot(z[flag,0], z[flag,1], ".", alpha = 0.2)
        plt.plot(z[~flag,0], z[~flag,1], ".", alpha = 0.2)
        plt.xlabel(r'$z_1$')
        plt.ylabel(r'$z_2$')
        plt.title("idx_step: {}".format(idx_step))
        z_from_x_plots.savefig()

x_from_z_plots.close()
z_from_x_plots.close()
