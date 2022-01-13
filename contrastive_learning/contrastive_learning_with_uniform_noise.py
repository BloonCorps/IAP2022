__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2021/01/09 19:34:07"

import numpy as np
from functions import *
from sys import exit
import argparse
from scipy.interpolate import BSpline
from scipy import optimize
import matplotlib as mpl
from matplotlib import cm


beta = 0.05
with open('../../output/range.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max, x2_min, x2_max = data['x1_min'], data['x1_max'], data['x2_min'], data['x2_max']

## test
# num_samples = 30
# x1 = np.random.rand(num_samples)*(x1_max - x1_min) + x1_min
# x2 = np.random.rand(num_samples)*(x2_max - x2_min) + x2_min
# x = np.vstack([x1, x2]).T
# y = compute_cubic_spline_basis(x)
# print(y.shape) # number of samples * 144 (12*12), 12 comes from 10+3(k)-1 of B-spline
#

## samples from p
with open("../output/TREMC/samples_beta_%.3f.pkl"%beta, 'rb') as file_handle:
    data = pickle.load(file_handle)
xp = np.array(data['x_record'])[:, -1, :] ## beta = 0.05
num_samples_p = xp.shape[0]

## samples from q
num_samples_q = num_samples_p
x1_q = np.random.rand(num_samples_q)*(x1_max - x1_min) + x1_min
x2_q = np.random.rand(num_samples_q)*(x2_max - x2_min) + x2_min
xq = np.vstack([x1_q, x2_q]).T

## coefficients of cubic splines
# theta = np.random.randn(144)
# F = np.zeros(1)

def compute_loss_and_grad(thetas):
    theta = thetas[0:144]
    F = thetas[-1]

    xp_basis = compute_cubic_spline_basis(xp)   # 20,000*144
    up_xp = np.matmul(xp_basis, theta)          # 20,000*144 * 144*1
    logp_xp = -(up_xp - F)
    logq_xp = np.ones_like(logp_xp)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    xq_basis = compute_cubic_spline_basis(xq)
    up_xq = np.matmul(xq_basis, theta)
    logp_xq = -(up_xq - F)
    logq_xq = np.ones_like(logp_xq)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    nu = num_samples_q / num_samples_p

    G_xp = logp_xp - logq_xp
    G_xq = logp_xq - logq_xq

    h_xp = 1./(1. + nu*np.exp(-G_xp))
    h_xq = 1./(1. + nu*np.exp(-G_xq))

    loss = -(np.mean(np.log(h_xp)) + nu*np.mean(np.log(1-h_xq)))

    ## why?
    dl_dtheta = -(np.mean((1 - h_xp)[:, np.newaxis]*(-xp_basis), 0) +
                  nu*np.mean(-h_xq[:, np.newaxis]*(-xq_basis), 0))
    dl_dF = -(np.mean(1 - h_xp) + nu*np.mean(-h_xq))

    return loss, np.concatenate([dl_dtheta, np.array([dl_dF])])



thetas_init = np.random.randn(145)
# loss, grad = compute_loss_and_grad(thetas_init)

thetas, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 thetas_init,
                                 iprint = 1)
#                                 factr = 10)
theta = thetas[0:144]
F = thetas[-1]

x = generate_grid(x1_min, x1_max, x2_min, x2_max)
basis = compute_cubic_spline_basis(x.numpy())       # 10,000*144

up = np.matmul(basis, theta)                        # 10,000*144 * 144*1 = 10,000*1
up = up.reshape(100, 100)
up = up.T

fig = plt.figure(0)
fig.clf()
plt.contourf(up, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.colorbar()
plt.tight_layout()
# plt.savefig("../output/learned_Up_beta_%.3f.pdf"%beta)
plt.show()


exit()
