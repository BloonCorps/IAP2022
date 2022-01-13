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
from scipy.optimize import fsolve
from scipy import optimize

## func to directly solve
## and Newton equation
# cutoff = 1E-5
# def func(deltaF, phiF, phiR, beta, n_samples):
#     deltaF = torch.tensor([deltaF]*n_samples)
#     res = float(torch.mean(1/(1+torch.exp(beta*(phiF-deltaF))))-torch.mean(1/(1+torch.exp(beta*(phiR+deltaF)))))
#     return res

# def func_prime(deltaF, phiF, phiR, beta, n_samples):
#     deltaF = torch.tensor([deltaF]*n_samples)
#     exp_a = torch.exp(phiF-deltaF)
#     exp_b = torch.exp(phiR+deltaF)
#     fermi_a = exp_a/((1+exp_a)**2)
#     fermi_b = exp_b/((1+exp_b)**2)
#     res = float(torch.mean(fermi_a)+torch.mean(fermi_b))
#     return res



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
with open('../output/TREMC_basin1/samples_beta_%.3f.pkl'%beta, 'rb') as file_handle:
    sample1 = pickle.load(file_handle)
xa = np.array(sample1['x_record'])[:, -1, :]
n_samples = xa.shape[0]
xa = torch.from_numpy(xa)

## get samples from b
with open('../output/TREMC_basin2/samples_beta_%.3f.pkl'%beta, 'rb') as file_handle:
    sample2 = pickle.load(file_handle)
xb = np.array(sample2['x_record'])[:, -1, :]
xb = torch.from_numpy(xb)




## func to use optimization
## use L_BFGS
def compute_func_and_grad(deltaF, phiF, phiR):
    fac_phiF = torch.exp(-beta*(phiF-torch.tensor([deltaF]*n_samples)))
    fac_phiR = torch.exp(-beta*(-phiR+torch.tensor([deltaF]*n_samples)))
    func = float(1/beta*(torch.mean(torch.log(1+fac_phiF))+torch.mean(torch.log(1+fac_phiR))))
    grad = float(torch.mean(fac_phiF/(1+fac_phiF))-torch.mean(fac_phiR/(1+fac_phiR)))
    return func, grad



## get the potential from contrastive learning
with open('../output/contrastive_learning_Up/learned_Up_beta%.3f_basin1_theta.pkl'%beta, 'rb') as file_handle:
    param1 = pickle.load(file_handle)
with open('../output/contrastive_learning_Up/learned_Up_beta%.3f_basin2_theta.pkl'%beta, 'rb') as file_handle:
    param2 = pickle.load(file_handle)
theta1, theta2 = param1['theta'], param2['theta']

## ua
uax = np.matmul(compute_cubic_spline_basis(xa.numpy()), theta1)
ubx = np.matmul(compute_cubic_spline_basis(xb.numpy()), theta2)

## extract flow from each saved step
num_steps = list(map(int, np.logspace(1.4,4.3, 50)))
deltaF_lst = []
np.random.seed(10)
# for i in tqdm(range(len(num_steps))):
# for i in range(len(num_steps)):
for i in range(1):
    idx_step = num_steps[i]
    ptfile = '../output/LBAR/model_beta%.3f_step%d.pt'%(beta, idx_step)
    model = torch.load(ptfile)
    realNVP.load_state_dict(model['state_dict'])

    with torch.no_grad():
        m_xa, logdetm = realNVP(xa)
        m_xb, logdetmm1 = realNVP.inverse(xb)
    ubmx = torch.tensor(np.matmul(compute_cubic_spline_basis(m_xa.numpy()), theta2))
    uamm1x = torch.tensor(np.matmul(compute_cubic_spline_basis(m_xb.numpy()), theta1))

    phiF = ubmx - uax - beta**(-1)*logdetm
    phiR = uamm1x - ubx - beta**(-1)*logdetmm1

    print("start to opt...")

    ## fsolve f(x) = 0
    # root = fsolve(func, deltaF0, args=(phiF, phiR, beta, n_samples), fprime=func_prime)

    ## newton optimization
    # deltaF_i = deltaF0    
    # while True:
    #     f_deltaFi = func(deltaF_i, phiF, phiR, beta, n_samples)
    #     f_deltaFi_prime = func_prime(deltaF_i, phiF, phiR, beta, n_samples)
    #     print(f_deltaFi, f_deltaFi_prime)
    #     deltaF_ip1 = deltaF_i - f_deltaFi/f_deltaFi_prime
    #     print(deltaF_ip1)
    #     if np.abs(deltaF_ip1-deltaF_i) < cutoff:
    #         break
    #     deltaF_i = deltaF_ip1

    ## use L_BFGS to optimize
    deltaF0 = np.random.randn()
    func, grad = compute_func_and_grad(deltaF0, phiF, phiR)
    deltaF, f, d = optimize.fmin_l_bfgs_b(compute_func_and_grad, x0=deltaF0, args=(phiF, phiR), iprint=1)
    # print(deltaF)


    # deltaF = (-1/beta)*torch.log(torch.mean(torch.exp(-beta*phiF)))
    # deltaF_lst.append(beta*deltaF)

## plotting
# fig = plt.figure()
# fig.clf()
# plt.plot(num_steps, deltaF_lst)
# plt.xscale("log")
# plt.xlabel(r"$step$", fontsize=20)
# plt.ylabel(r"$\beta \Delta F$", fontsize=20)
# plt.tight_layout()
# plt.savefig('../output/LBAR/deltaF_LFEP_basin1_beta%.3f.pdf'%beta)

sys.exit()


