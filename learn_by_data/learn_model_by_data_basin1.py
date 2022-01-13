__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 21:50:25"

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import torch.distributions as distributions
import pickle
import math
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from functions import *
import argparse
from RealNVP import *
from tqdm import tqdm

beta = 0.05
with open('../output/range.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max, x2_min, x2_max = data['x1_min'], data['x1_max'], data['x2_min'], data['x2_max']

## Masks used to define the number and the type of affine coupling layers
## In each mask, 1 means that the variable at the correspoding position is
## kept fixed in the affine couling layer

masks = [("Affine Coupling", [1.0, 0.0]), ("Affine Coupling", [0.0, 1.0])]*3
masks.append(("Scale", [0.0, 1.0], x1_min, x1_max))
masks.append(("Scale", [1.0, 0.0], x2_min, x2_max))

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 16

## construct the RealNVP_2D object
realNVP = RealNVP(masks, hidden_dim)

if torch.cuda.device_count():
    realNVP = realNVP.cuda()

with open("../output/TREMC_basin1/samples_beta_{:.3f}.pkl".format(beta), 'rb') as file_handle:
    data = pickle.load(file_handle)
    x_record, beta_lst = data['x_record'], data['beta_lst']

x_record = np.array(x_record)
x = x_record[:, -1, :] ## beta = 0.05
num_samples = x.shape[0]
num_train_samples = int(0.8*num_samples)
train_sample_idx = np.random.choice(range(num_samples), size = num_train_samples, replace=False)
train_sample_flag = [False]*num_samples
for idx in train_sample_idx:
    train_sample_flag[idx] = True

train_sample_flag = np.array(train_sample_flag)
x_train = x[train_sample_flag]
x_validation = x[~train_sample_flag]

optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)
num_saved_steps = np.logspace(1.4, 4.3, 50)
num_saved_steps = list(num_saved_steps.astype(np.int32))

## learn the RealNVP model
loss_train_record = []
loss_validation_record = []
x_train = torch.from_numpy(x_train)
x_train = x_train.to(next(realNVP.parameters()).device)

x_validation = torch.from_numpy(x_validation)
x_validation = x_validation.to(next(realNVP.parameters()).device)

normal_dist = distributions.Normal(0.0, 1.0)

for idx_step in tqdm(range(num_saved_steps[-1] + 1)):
    z, logdet = realNVP.inverse(x_train)
    loss = -torch.mean(torch.sum(normal_dist.log_prob(z), -1) + logdet)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_train_record.append(loss.item())
    
    if idx_step in num_saved_steps:
        with torch.no_grad():
            z, logdet = realNVP.inverse(x_validation)
        loss_validation = -torch.mean(torch.sum(normal_dist.log_prob(z), -1) + logdet)
        loss_validation_record.append(loss_validation.item())
        
        torch.save({'masks': masks,
                    'hidden_dim': hidden_dim,
                    'beta': beta,
                    'loss_train': loss.item(),
                    'loss_validation': loss_validation.item(),
                    'state_dict': realNVP.state_dict()},
                    "../output/model_trained_with_data_basin1/model_beta_{:.3f}_step_{}.pt".format(beta, idx_step))

        # print(f"idx_steps: {idx_step:}, loss_train: {loss.item():.3f}, loss_validation: {loss_validation.item():.3f}")

fig = plt.figure(0)
fig.clf()
plt.plot(loss_train_record)
fig.savefig("../output/loss_model_trained_with_data_basin1_beta_{:.3f}.pdf".format(beta))
