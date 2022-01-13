__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/04 17:27:27"

import numpy as np
import torch
torch.set_default_dtype(torch.double)
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
from matplotlib import cm
from sys import exit
import pickle

'''
The Muller potential functions, U(x), is defined.
The corresponding probability densities are defined as \log P(x) \propto \exp(-U(x))

'''

def compute_Muller_potential(beta, x):
    A = (-200., -100., -170., 15.)
    b = (0., 0., 11., 0.6)    
    ac = (x.new_tensor([-1.0, -10.0]),
          x.new_tensor([-1.0, -10.0]),
          x.new_tensor([-6.5, -6.5]),
          x.new_tensor([0.7, 0.7]))
    
    x0 = (x.new_tensor([ 1.0, 0.0]),
          x.new_tensor([ 0.0, 0.5]),
          x.new_tensor([-0.5, 1.5]),
          x.new_tensor([-1.0, 1.0]))
    
    U = 0    
    for i in range(4):
        diff = x - x0[i]
        U = U + A[i]*torch.exp(torch.sum(ac[i]*diff**2, -1) + b[i]*torch.prod(diff, -1))

    U = beta * U
    return U
    

def generate_grid(x1_min, x1_max, x2_min, x2_max, ndim1, ndim2):
    x1 = torch.linspace(x1_min, x1_max, steps=ndim1)
    x2 = torch.linspace(x2_min, x2_max, steps=ndim2)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2)
    grid = torch.stack([grid_x1, grid_x2], dim = -1)
    x = grid.reshape((-1, 2))
    return x


if __name__ == "__main__":

    ## muller potential boundary
    x1_min, x1_max = -1.5, 1.0
    x2_min, x2_max = -0.5, 2.0
    ndim = 100

    x = generate_grid(x1_min, x1_max, x2_min, x2_max, ndim, ndim)
    beta = 0.05
    U = compute_Muller_potential(beta, x)
    U = U.reshape(ndim, ndim)
    U[U>15] = 15
    U = U.T

    with open('../output/range.pkl', 'wb') as file_handle:
        pickle.dump({'x1_min': x1_min, 'x1_max': x1_max,
                     'x2_min': x2_max, 'x2_max': x2_max,
                     'U': U}, file_handle)


    fig = plt.figure(0)
    fig.clf()
    plt.contourf(U, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
    plt.xlabel(r"$x_1$", fontsize = 24)
    plt.ylabel(r"$x_2$", fontsize = 24)
    plt.colorbar()
    plt.tight_layout()
    # plt.savefig("../output/true_muller_energy_beta_%.3f.eps"%beta)
