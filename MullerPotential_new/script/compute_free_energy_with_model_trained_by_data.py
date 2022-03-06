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
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from RealNVP import *
from functions import *
from FastMBAR import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

num_saved_steps = np.logspace(1.4, 4.3, 50)
num_saved_steps = list(num_saved_steps.astype(np.int32))
num_saved_steps = num_saved_steps[0:40]

data = torch.load("./output/model_trained_by_data/model_alpha_{:.3f}_step_{}.pt".format(alpha, num_saved_steps[0]))
hidden_dim = data['hidden_dim']
masks = data['masks']
x1_min, x1_max = data['masks'][-2][2], data['masks'][-2][3]
x2_min, x2_max = data['masks'][-1][2], data['masks'][-1][3]

with open("./output/F_numeric_integration_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    Fp = data['F']

realNVP = RealNVP(masks, hidden_dim)
#realNVP.cuda()

with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    x_record = data['x_record']
    alphas = data['alphas']    
x_train = x_record[:, -1, :]
x_train = torch.from_numpy(x_train)
x_train = x_train.to(next(realNVP.parameters()).device)

## compute DKL(p(x)||q(x))
Fp_DKL_pq_repeats = []
Fp_DKL_qp_repeats = []
Fp_BAR_repeats = []
num_steps_saved = []
for idx_step in num_saved_steps:
    Fp_DKL_pq = []
    Fp_DKL_qp = []
    Fp_BAR = []
    
    print("idx_step: {}".format(idx_step))

    data = torch.load("./output/model_trained_by_data/model_alpha_{:.3f}_step_{}.pt".format(alpha, idx_step))
    realNVP.load_state_dict(data['state_dict'])

    normal_dist = distributions.Normal(0.0, 1.0)        
    num_steps_saved.append(idx_step)        
    for r in range(5):
        indices = np.random.choice(x_train.shape[0], x_train.shape[0], replace = True)
        x_sample = x_train[indices]
        energy = compute_Muller_potential(alpha, x_sample)
        p_logp = -energy

        with torch.no_grad():
            z, logdet = realNVP.inverse(x_train)
            p_logq = torch.sum(normal_dist.log_prob(z), -1) + logdet

        z_sample = normal_dist.sample([x_train.shape[0], 2])
        z_sample = z_sample.to(next(realNVP.parameters()).device)
        with torch.no_grad():
            x_sample, logdet = realNVP(z_sample)            
        q_logp = -compute_Muller_potential(alpha, x_sample)
        q_logq = torch.sum(normal_dist.log_prob(z_sample), -1) - logdet

        Fp_DKL_pq.append(-torch.mean(p_logp - p_logq).item())
        Fp_DKL_qp.append(torch.mean(q_logq - q_logp).item())            

        energy_matrix = (-1) * torch.stack([torch.cat([q_logq, p_logq]),
                                            torch.cat([q_logp, p_logp])],
                                           dim = 0).numpy()
        num_conf = np.array([len(q_logp), len(p_logp)])
        fastmbar = FastMBAR(energy_matrix, num_conf, verbose = False)
        Fp_BAR.append(fastmbar.F[-1])

    Fp_DKL_pq_repeats.append(Fp_DKL_pq)
    Fp_DKL_qp_repeats.append(Fp_DKL_qp)
    Fp_BAR_repeats.append(Fp_BAR)

Fp_DKL_qp_repeats = np.array(Fp_DKL_qp_repeats)
Fp_DKL_pq_repeats = np.array(Fp_DKL_pq_repeats)
Fp_BAR_repeats = np.array(Fp_BAR_repeats)

exit()


fig = plt.figure(2)
fig.clf()
p1, = plt.plot(num_saved_steps, np.mean(Fp_DKL_pq_repeats, -1), color = 'orange')
p2, = plt.plot(num_saved_steps, np.mean(Fp_DKL_qp_repeats, -1), color = 'green')
p3, = plt.plot(num_saved_steps, np.mean(Fp_BAR_repeats, -1), color = 'blue')
p4 = plt.axhline(y = Fp, color = "k")

plt.legend([(p4), (p3), (p2), (p1)], ["Exact", "BAR", r"$\leftangle -W_{A \rightarrow A^{\circ}} \rightangle$", r"$\leftangle W_{A^{\circ} \rightarrow A} \rightangle$"],
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, ncol = 2)
plt.ylim([-7.5,-1.0])
plt.xlabel("Number of training steps")
plt.xscale('log')
plt.tight_layout()
plt.savefig("./output/analyze_model_trained_by_data/F.eps")

exit()
