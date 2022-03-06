__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 21:50:25"

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from RealNVP import *
from functions import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

## Masks used to define the number and the type of affine coupling layers
## In each mask, 1 means that the variable at the correspoding position is
## kept fixed in the affine couling layer

masks = [("Affine Coupling", [1.0, 0.0]), ("Affine Coupling", [0.0, 1.0])]*3
x1_min, x1_max = -1.5, 1.0
x2_min, x2_max = -0.5, 2.0
with open("./output/range.pkl", 'wb') as file_handle:
    pickle.dump({'x1_min': x1_min,
                 'x1_max': x1_max,
                 'x2_min': x2_min,
                 'x2_max': x2_max}, file_handle)

masks.append(("Scale", [0.0, 1.0], x1_min, x1_max))
masks.append(("Scale", [1.0, 0.0], x2_min, x2_max))

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 56

## construct the RealNVP_2D object
realNVP = RealNVP(masks, hidden_dim)

if torch.cuda.device_count():
    realNVP = realNVP.cuda()
    
optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)
num_steps = 1000

## learn the RealNVP model
loss_record = []
for idx_step in range(num_steps):
    Z = torch.normal(0, 1, size = (1024, 2))
    Z = Z.cuda()
    X, logdet = realNVP(Z)

    logp = -compute_Muller_potential(alpha, X)
    loss = torch.mean(-logdet - logp)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_record.append(loss.item())
    
    if (idx_step + 1) % 10 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")

    if (idx_step + 1) % 25 == 0:
        torch.save({'masks': masks,
                    'hidden_dim': hidden_dim,
                    'alpha': alpha,
                    'loss_record': loss_record,
                    'state_dict': realNVP.state_dict()},
                   "./output/model_trained_by_potential/model_alpha_{:.3f}_step_{}.pt".format(alpha, idx_step))
        
fig = plt.figure(0)
fig.clf()
plt.plot(loss_record)
fig.savefig("./output/loss_model_trained_by_potential_alpha_{:.3f}.pdf".format(alpha))
