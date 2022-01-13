import sys
sys.path.append('../')
from RealNVP import *
from functions import *
import numpy as np
import pickle
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import matplotlib as mpl
from matplotlib import cm
from tqdm import tqdm

## basic parameters
beta = 0.05
with open('../output/range.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max, x2_min, x2_max = data['x1_min'], data['x1_max'], data['x2_min'], data['x2_max']


## Masks used to define the number and the type of affine coupling layers
## In each mask, 1 means that the variable at the correspoding position is
## kept fixed in the affine couling layer
masks = [("Scale_inverse", [0.0,1.0], x1_min, x1_max), ("Scale_inverse", [1.0,0.0], x2_min, x2_max)]
masks += [("Affine Coupling", [1.0, 0.0]), ("Affine Coupling", [0.0, 1.0])]*3
masks.append(("Scale", [0.0, 1.0], x1_min, x1_max))
masks.append(("Scale", [1.0, 0.0], x2_min, x2_max))

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 16
with open('../output/LFEP/nn_architect_beta%.3f.pkl'%beta, 'wb') as file_handle:
    pickle.dump({'masks':masks, 'hidden_dim': hidden_dim}, file_handle)


## contruct the realNVP object
realNVP = RealNVP(masks, hidden_dim)
if torch.cuda.device_count():
    realNVP = realNVP.cuda()


## get the samples from state a
with open('../output/TREMC_basin1/samples_beta_%.3f.pkl'%beta, 'rb') as file_handle:
    sample1 = pickle.load(file_handle)
with open('../output/TREMC_basin2/samples_beta_%.3f.pkl'%beta, 'rb') as file_handle:
    sample2 = pickle.load(file_handle)
xa = np.array(sample1['x_record'])[:, -1, :]
xb = np.array(sample2['x_record'])[:, -1, :]
n_samples = xa.shape[0]
if n_samples != xb.shape[0]:
    print('Samples are not equal')
    sys.exit()

# (get some validation samples)
n_train = int(0.8*n_samples)
train_sample_index = np.random.choice(range(n_samples), size=n_train, replace=False)
train_sample_flag = np.array([True if idx in train_sample_index else False for idx in range(n_samples)])
xa_train, xb_train = xa[train_sample_flag], xb[train_sample_flag]
xa_valid, xb_valid = xa[~train_sample_flag], xb[~train_sample_flag]

# (change the training data to torch format)
xa_train, xb_train = torch.from_numpy(xa_train), torch.from_numpy(xb_train)
xa_train, xb_train = xa_train.to(next(realNVP.parameters()).device), xb_train.to(next(realNVP.parameters()).device)


## get the potential from contrastive learning
with open('../output/contrastive_learning_Up/learned_Up_beta%.3f_basin1_theta.pkl'%beta, 'rb') as file_handle:
    param1 = pickle.load(file_handle)
with open('../output/contrastive_learning_Up/learned_Up_beta%.3f_basin2_theta.pkl'%beta, 'rb') as file_handle:
    param2 = pickle.load(file_handle)
theta1, theta2 = param1['theta'], param2['theta']

## ua and ub
uax = np.matmul(compute_cubic_spline_basis(xa_train.numpy()), theta1)
ubx = np.matmul(compute_cubic_spline_basis(xb_train.cpu().numpy()), theta2)
uax_valid = np.matmul(compute_cubic_spline_basis(xa_valid), theta1)
ubx_valid = np.matmul(compute_cubic_spline_basis(xb_valid), theta2)


## define optimizer
optimizer = optim.Adam(realNVP.parameters(), lr=1e-4)
num_steps = list(map(int, np.logspace(1.4,4.3, 50)))

## training process
loss_train_record = []
loss_valid_record = []

for idx_step in tqdm(range(num_steps[-1] + 1)):
# for idx_step in range(num_steps[-1] + 1):
#     if idx_step%1000 == 0:
#         print('idx_step: %d ....\n'%idx_step)

    # m_xa, logdetm = realNVP(xa_train)
    m_xb, logdetmm1 = realNVP.inverse(xb_train)

    # ubmx = torch.tensor(np.matmul(compute_cubic_spline_basis(m_xa.detach().numpy()), theta2))
    uamm1x = torch.tensor(np.matmul(compute_cubic_spline_basis(m_xb.detach().numpy()), theta1))

    # phiF = ubmx - uax - beta**(-1)*logdetm
    phiR = uamm1x - ubx - beta**(-1)*logdetmm1

    # loss = torch.mean(phiF)
    loss = torch.mean(phiR)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_train_record.append(loss.item())

    if idx_step in num_steps:
        with torch.no_grad():
            m_xb, logdetmm1 = realNVP.inverse(torch.from_numpy(xb_valid))
            uamm1x = torch.tensor(np.matmul(compute_cubic_spline_basis(m_xb.numpy()), theta1))
            loss_valid = torch.mean(uamm1x - ubx_valid - beta**(-1)*logdetmm1)
            loss_valid_record.append(loss_valid)
        torch.save({'loss_train': loss.item(),
                    'loss_valid': loss_valid.item(),
                    'state_dict': realNVP.state_dict()},
                    '../output/LFEP/model_basin2_beta%.3f_step%d.pt'%(beta, idx_step))

## store training and validation loss
with open('../output/LFEP/loss_record_basin2_beta%.3f.pkl'%beta, 'wb') as file_handle:
	pickle.dump({'training_loss': loss_train_record, 'validation_loss':loss_valid_record}, file_handle)

sys.exit()
