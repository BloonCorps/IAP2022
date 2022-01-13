__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/12/14 02:42:51"

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
torch.set_default_dtype(torch.double)
import pickle
import argparse
import sys
from functions import *
import torch
from tqdm import tqdm

beta = 0.05
with open('../../output/range.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max, x2_min, x2_max = data['x1_min'], data['x1_max'], data['x2_min'], data['x2_max']

## temperate replica
num_reps = 11
beta_lst = torch.linspace(0, beta, num_reps)
x0_1 = x1_min+np.random.rand(num_reps)*(x1_max-x1_min)
x0_2 = x2_min+np.random.rand(num_reps)*(x2_max-x2_min)
x = np.column_stack((x0_1, x0_2))
U = compute_Muller_potential(1.0, torch.from_numpy(x))

nsteps = 210000
record = []

## sampling
for step in tqdm(range(nsteps), bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}'):
    # if (step+1) % 100 == 0:
    #     print('step index: %d'%step)

    ## mcmc
    delta_x = np.random.normal(0, 1, size=(num_reps, 2))*0.3
    x_candidate = x + delta_x
    U_candidate = compute_Muller_potential(1.0, torch.from_numpy(x_candidate))

    accept_prob = torch.exp(-beta_lst*(U_candidate-U))
    accept_bool = torch.rand(num_reps) < accept_prob


    x1_min_1, x1_max_1 = -0.8, x1_max
    x2_min_1, x2_max_1 = x2_min, 0.8
    # x1_bar, x2_bar = 0.5*(x1_min_1+x1_max_1), 0.5*(x2_min_1+x2_max_1)
    x1_bar, x2_bar = x1_min_1, x2_max_1
    cutoff = -0.025
    for i, xi in enumerate(x_candidate):
        booli = all(xi > [x1_min_1, x2_min_1]) and \
                all(xi < [x1_max_1, x2_max_1]) and \
                (xi[0]-x1_bar)*(xi[1]-x2_bar) < cutoff
        if not booli:
            accept_bool[i] = False

    x_candidate[~accept_bool] = x[~accept_bool]
    U_candidate[~accept_bool] = U[~accept_bool]
    x = x_candidate
    U = U_candidate

    ## temperature replica exchange
    if (step+1) % 10 == 0:
        for itemp in range(1, num_reps):
            accept_prob = torch.exp((beta_lst[itemp]-beta_lst[itemp-1])*(U[itemp]-U[itemp-1]))
            accept_bool = torch.rand(1) < accept_prob
            if accept_bool.item():
                x[itemp], x[itemp-1] = x[itemp-1], x[itemp]
                U[itemp], U[itemp-1] = U[itemp-1], U[itemp]
        
        if step > 10000:
            record.append(x)

with open('../output/TREMC_basin2/samples_beta_%.3f.pkl'%beta, 'wb') as file_handle:
    pickle.dump({'beta_lst': beta_lst, 'x_record': record}, file_handle)


## load data directly
U = data['U']
with open('../output/TREMC_basin2/samples_beta_%.3f.pkl'%beta, 'rb') as file_handle:
    data = pickle.load(file_handle)
record = data['x_record']
record = np.array(record)


## plotting all the data
for i in range(num_reps):
    fig = plt.figure(i)
    fig.clf()
    plt.plot(record[:, i, 0], record[:, i, 1], '.', markersize=2, alpha = 0.5)
    plt.contourf(U, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title('beta = %.3f'%beta_lst[i])
    plt.tight_layout()
    plt.savefig("../output/TREMC_basin2/%.3f.pdf"%beta_lst[i])

