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
import torch
torch.set_default_dtype(torch.float64)
import torch.distributions as distributions
from RealNVP import *

beta = 0.05
with open('../output/range.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max, x2_min, x2_max = data['x1_min'], data['x1_max'], data['x2_min'], data['x2_max']


data = torch.load("../output/model_trained_with_data/model_beta_{:.3f}_step_19952.pt".format(beta), map_location=torch.device('cpu'))
hidden_dim = data['hidden_dim']
masks = data['masks']
realNVP = RealNVP(masks, hidden_dim)
realNVP.load_state_dict(data['state_dict'])

## samples from p
with open("../output/TREMC/samples_beta_{:.3f}.pkl".format(beta), 'rb') as file_handle:
    data = pickle.load(file_handle)
xp = np.array(data['x_record'])[:, -1, :]
num_samples_p = xp.shape[0]

## samples from q
normal_dist = torch.distributions.Normal(0.0, 2.0)            
num_samples_q = num_samples_p
zq = normal_dist.sample((num_samples_q, 2))
with torch.no_grad():
    xq, logdet = realNVP(zq)
logq_xq = torch.sum(normal_dist.log_prob(zq), -1) - logdet
    
xq = xq.cpu().detach().numpy()
logq_xq = logq_xq.cpu().detach().numpy()

with torch.no_grad():
    zp, logdet = realNVP.inverse(torch.from_numpy(xp))
    logq_xp = torch.sum(normal_dist.log_prob(zp), -1) + logdet
logq_xp = logq_xp.cpu().detach().numpy()

## coefficients of cubic splines
theta = np.random.randn(144)
F = np.zeros(1)

xp_basis = compute_cubic_spline_basis(xp)
xq_basis = compute_cubic_spline_basis(xq)

def compute_loss_and_grad(thetas):
    theta = thetas[0:144]
    F = thetas[-1]

    #xp_basis = compute_cubic_spline_basis(xp)
    up_xp = np.matmul(xp_basis, theta)
    logp_xp = -(up_xp - F)
    #logq_xp = np.ones_like(logp_xp)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    #xq_basis = compute_cubic_spline_basis(xq)
    up_xq = np.matmul(xq_basis, theta)
    logp_xq = -(up_xq - F)
    #logq_xq = np.ones_like(logp_xq)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    nu = num_samples_q / num_samples_p

    G_xp = logp_xp - logq_xp
    G_xq = logp_xq - logq_xq

    h_xp = 1./(1. + nu*np.exp(-G_xp))
    h_xq = 1./(1. + nu*np.exp(-G_xq))

    loss = -(np.mean(np.log(h_xp)) + nu*np.mean(np.log(1-h_xq)))

    dl_dtheta = -(np.mean((1 - h_xp)[:, np.newaxis]*(-xp_basis), 0) +
                  nu*np.mean(-h_xq[:, np.newaxis]*(-xq_basis), 0))
    dl_dF = -(np.mean(1 - h_xp) + nu*np.mean(-h_xq))

    return loss, np.concatenate([dl_dtheta, np.array([dl_dF])])

thetas_init = np.random.randn(145)
loss, grad = compute_loss_and_grad(thetas_init)

thetas, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 thetas_init,
                                 iprint = 1)
#                                 factr = 10)
theta = thetas[0:144]
F = theta[-1]

x = generate_grid(x1_min, x1_max, x2_min, x2_max)
basis = compute_cubic_spline_basis(x.numpy())

up = np.matmul(basis, theta)
up = up.reshape(100, 100)
up = up.T

fig = plt.figure(0)
fig.clf()
plt.contourf(up, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.colorbar()
plt.tight_layout()
plt.savefig("../output/learned_Up_beta_{:.3f}_with_flow.pdf".format(beta))


normal_dist = torch.distributions.Normal(0.0, 1.0)            
num_samples_q = num_samples_p
zq = normal_dist.sample((num_samples_q, 2))
with torch.no_grad():
    xq, logdet = realNVP(zq)
    
fig = plt.figure(1)
fig.clf()
plt.plot(xq[:,0], xq[:, 1], '.')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig("../output/sample_from_q_beta_{:.3f}_with_flow.pdf".format(beta))

exit()

# normal_dist = torch.distributions.Normal(0.0, 1.0)
# # z = normal_dist.sample((1000, 2))
# # with torch.no_grad():
# #     x, logdet = realNVP(z)

# with torch.no_grad():
#     z, logdet = realNVP.inverse(x)
#     logq_x = torch.sum(normal_dist.log_prob(z), -1) + logdet
# logq_x = logq_x.cpu().detach().numpy()
# logq_x = logq_x.reshape(100, 100)
# logq_x = logq_x.T

# Uq = -logq_x
# Uq = Uq - np.min(Uq)
# Uq[Uq > 30] = np.nan

# fig = plt.figure(0)
# fig.clf()
# plt.contourf(Uq, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
# plt.xlabel(r"$x_1$", fontsize = 24)
# plt.ylabel(r"$x_2$", fontsize = 24)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig("./output/Uq_alpha_{:.3f}_from_flow.pdf".format(alpha))

# exit()
