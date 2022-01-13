import sys
sys.path.append('../')
from RealNVP import *
from functions import *
import numpy as np
import pickle
import torch
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt
from tqdm import tqdm

## basic parameters
beta = 0.05
with open('../output/range.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max, x2_min, x2_max = data['x1_min'], data['x1_max'], data['x2_min'], data['x2_max']


## model architect
with open('../output/LFEP/nn_architect_beta%.3f.pkl'%beta, 'rb') as file_handle:
    nn = pickle.load(file_handle)
masks, hidden_dim = nn['masks'], nn['hidden_dim']
realNVP = RealNVP(masks, hidden_dim)


## get samples from a
with open('../output/TREMC_basin2/samples_beta_%.3f.pkl'%beta, 'rb') as file_handle:
    sample2 = pickle.load(file_handle)
xb = np.array(sample2['x_record'])[:, -1, :]
n_samples = xb.shape[0]
xb = torch.from_numpy(xb)


## get the potential from contrastive learning
with open('../output/contrastive_learning_Up/learned_Up_beta%.3f_basin1_theta.pkl'%beta, 'rb') as file_handle:
    param1 = pickle.load(file_handle)
with open('../output/contrastive_learning_Up/learned_Up_beta%.3f_basin2_theta.pkl'%beta, 'rb') as file_handle:
    param2 = pickle.load(file_handle)
theta1, theta2 = param1['theta'], param2['theta']


## ub
ubx = np.matmul(compute_cubic_spline_basis(xb.numpy()), theta2)


## extract flow from each saved step
num_steps = list(map(int, np.logspace(1.4,4.3,50)))
deltaF_lst = []
for i in tqdm(range(len(num_steps))):
    idx_step = num_steps[i]
    ptfile = '../output/LFEP/model_basin2_beta%.3f_step%d.pt'%(beta, idx_step)
    model = torch.load(ptfile)
    realNVP.load_state_dict(model['state_dict'])

    with torch.no_grad():
        m_xb, logdetmm1 = realNVP.inverse(xb)
    uamm1x = torch.tensor(np.matmul(compute_cubic_spline_basis(m_xb.numpy()), theta1))

    phiR = uamm1x - ubx - beta**(-1)*logdetmm1
    deltaF = (-1/beta)*torch.log(torch.mean(torch.exp(-beta*phiR)))
    print(deltaF)
    deltaF_lst.append(beta*deltaF)


## plotting
fig = plt.figure()
fig.clf()
plt.plot(num_steps, deltaF_lst)
plt.xscale("log")
plt.xlabel(r"step")
plt.ylabel(r"$\beta \Delta F$")
plt.tight_layout()
# plt.show()
plt.savefig('../output/LFEP/deltaF_LFEP_basin2_beta%.3f.pdf'%beta)

sys.exit()


