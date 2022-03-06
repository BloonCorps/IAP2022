__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/25 18:54:04"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import pickle
import argparse
from matplotlib import cm
from scipy.stats import multivariate_normal
import scipy.integrate as integrate
from sys import exit
import sys
sys.path.append("./script")
from functions import *
import torch
torch.set_default_dtype(torch.double)
from FastMBAR import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open("./output/F_numeric_integration_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    Fp = data['F']

## read range
with open("./output/range.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max = data['x1_min'], data['x1_max']
x2_min, x2_max = data['x2_min'], data['x2_max']
U = data['U']

with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
    x_record = data['x_record']
    alphas = data['alphas']    
x = x_record[::, -1, :]
exit()

plots_samples = PdfPages("./output/mixture_of_normal/samples_alpha_{:.3f}.pdf".format(alpha))

#### normal distribution with one component ####
################################################
x_mean = np.mean(x, 0)
x_cov = np.cov(x.T)
dist = multivariate_normal(x_mean, x_cov)

def compute_density(x2, x1, dist):
    return dist.pdf((x1, x2))
Z, Z_error = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist],
                               epsabs = 1e-12, epsrel = 1e-12)
Fq = -np.log(Z)

Fp_DKL_pq_one_repeats = []
Fp_DKL_qp_one_repeats = []
Fp_BAR_one_repeats = []

for idx_rep in range(5):
    num_samples = x.shape[0]*20
    samples = dist.rvs(num_samples)
    flag = (samples[:,0] >= x1_min) & (samples[:,0] <= x1_max) &\
           (samples[:,1] >= x2_min) & (samples[:,1] <= x2_max)  
    samples = samples[flag]

    idx = np.random.choice(range(x.shape[0]), size = (x.shape[0],))
    x_resample = x[idx,:]
    p_logp = -compute_Muller_potential(alpha, torch.from_numpy(x_resample)).numpy()
    p_logq = dist.logpdf(x_resample)

    q_logp = -compute_Muller_potential(alpha, torch.from_numpy(samples)).numpy()
    q_logq = dist.logpdf(samples)

    Fp_DKL_pq_one = -np.mean(p_logp - p_logq) + Fq
    Fp_DKL_qp_one = np.mean(q_logq - q_logp) + Fq

    energy_matrix = (-1)*np.stack([np.concatenate([q_logq, p_logq]),
                                   np.concatenate([q_logp, p_logp])],
                                  0)
    num_conf = np.array([q_logq.shape[0], p_logq.shape[0]])
    fastmbar = FastMBAR(energy_matrix, num_conf, verbose = False)
    Fp_BAR_one = fastmbar.F[-1] + Fq

    Fp_DKL_pq_one_repeats.append(Fp_DKL_pq_one)
    Fp_DKL_qp_one_repeats.append(Fp_DKL_qp_one)
    Fp_BAR_one_repeats.append(Fp_BAR_one)

fig = plt.figure(0)
fig.clf()
plt.contourf(U, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
idx = np.random.choice(range(samples.shape[0]), size = 1000)
plt.plot(samples[:,0][idx], samples[:,1][idx], '.')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.colorbar()
plots_samples.savefig()

#### normal distribution with two components ####
#################################################
x_mean_0 = np.copy(x_mean)
x_mean_1 = np.copy(x_mean)

x_cov_0 = np.copy(x_cov)
x_cov_1 = np.copy(x_cov)

alpha_0 = 0.55
alpha_1 = 1 - alpha_0

## EM algorithm
curren_log_prob = -np.inf
stop_criteria = 1e-8
idx_step = 0
parameters = [{'x_mean_0': x_mean_0, 'x_mean_1': x_mean_1,
               'x_cov_0': x_cov_0, 'x_cov_1': x_cov_1,
               'alpha_0': alpha_0, 'alpha_1': alpha_1}]
while True:
    ## EM algorithm
    dist_0 = multivariate_normal(x_mean_0, x_cov_0)
    dist_1 = multivariate_normal(x_mean_1, x_cov_1)
    Z_0, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_0],
                               epsabs = 1e-12, epsrel = 1e-12)
    Z_1, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_1],
                               epsabs = 1e-12, epsrel = 1e-12)

    logp_0 = dist_0.logpdf(x) - np.log(Z_0) + np.log(alpha_0)
    logp_1 = dist_1.logpdf(x) - np.log(Z_1) + np.log(alpha_1)

    p_0 = np.exp(logp_0)
    p_1 = np.exp(logp_1)

    w_0 = p_0 / (p_0 + p_1)
    w_1 = 1 - w_0

    x_mean_0 = np.sum(x * w_0[:,np.newaxis], 0)/np.sum(w_0)
    x_mean_1 = np.sum(x * w_1[:,np.newaxis], 0)/np.sum(w_1)

    x_cov_0 = np.matmul(((x - x_mean_0)*w_0[:, np.newaxis]).T, (x - x_mean_0)*w_0[:, np.newaxis])/np.sum(w_0)
    x_cov_1 = np.matmul(((x - x_mean_1)*w_1[:, np.newaxis]).T, (x - x_mean_1)*w_1[:, np.newaxis])/np.sum(w_1)

    alpha_0 = np.sum(w_0)/(np.sum(w_0) + np.sum(w_1))
    alpha_1 = 1 - alpha_0

    log_prob = np.mean(np.log(p_0 + p_1))
    print("idx_step: {}, log_prob: {:2f}".format(idx_step, log_prob))

    parameters.append({'x_mean_0': x_mean_0, 'x_mean_1': x_mean_1,
                       'x_cov_0': x_cov_0, 'x_cov_1': x_cov_1,
                       'alpha_0': alpha_0, 'alpha_1': alpha_1})
    
    if log_prob - curren_log_prob < stop_criteria and idx_step > 10:
        break
    else:
        idx_step += 1
        curren_log_prob = log_prob

Fp_DKL_pq_two_repeats = []
Fp_DKL_qp_two_repeats = []
Fp_BAR_two_repeats = []
        
for idx_step in range(len(parameters)):
    print(idx_step)
    x_mean_0 = parameters[idx_step]['x_mean_0']
    x_mean_1 = parameters[idx_step]['x_mean_1']

    x_cov_0 = parameters[idx_step]['x_cov_0']
    x_cov_1 = parameters[idx_step]['x_cov_1']

    alpha_0 = parameters[idx_step]['alpha_0']
    alpha_1 = parameters[idx_step]['alpha_1']    
    
    dist_0 = multivariate_normal(x_mean_0, x_cov_0)
    dist_1 = multivariate_normal(x_mean_1, x_cov_1)
    
    Z_0, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_0],
                               epsabs = 1e-12, epsrel = 1e-12)
    Z_1, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_1],
                               epsabs = 1e-12, epsrel = 1e-12)

    Fp_DKL_pq_two = []
    Fp_DKL_qp_two = []
    Fp_BAR_two = []
    
    for idx_repeat in range(5):
        print("idx_repeat:", idx_repeat)
        

        idx = np.random.choice(range(x.shape[0]), size = (x.shape[0],))
        x_resample = x[idx,:]
        
        logp_0 = dist_0.logpdf(x_resample) - np.log(Z_0) + np.log(alpha_0)
        logp_1 = dist_1.logpdf(x_resample) - np.log(Z_1) + np.log(alpha_1)

        p_0 = np.exp(logp_0)
        p_1 = np.exp(logp_1)

        p_logq = np.log(p_0 + p_1)
        p_logp = -compute_Muller_potential(alpha, torch.from_numpy(x_resample)).numpy()

        num_samples = x.shape[0] * 20
        samples_0 = dist_0.rvs(num_samples)
        samples_1 = dist_1.rvs(num_samples)    

        flag = (samples_0[:,0] >= x1_min) & (samples_0[:,0] <= x1_max) &\
               (samples_0[:,1] >= x2_min) & (samples_0[:,1] <= x2_max)  
        samples_0 = samples_0[flag]

        flag = (samples_1[:,0] >= x1_min) & (samples_1[:,0] <= x1_max) &\
               (samples_1[:,1] >= x2_min) & (samples_1[:,1] <= x2_max)  
        samples_1 = samples_1[flag]

        num_samples_0 = int(num_samples*alpha_0)
        num_samples_1 = int(num_samples*alpha_1)

        samples_0 = samples_0[0:num_samples_0]
        samples_1 = samples_1[0:num_samples_1]    

        samples = np.concatenate([samples_0, samples_1])
        logp_0 = dist_0.logpdf(samples) - np.log(Z_0) + np.log(alpha_0)
        logp_1 = dist_1.logpdf(samples) - np.log(Z_1) + np.log(alpha_1)

        p_0 = np.exp(logp_0)
        p_1 = np.exp(logp_1)

        q_logq = np.log(p_0 + p_1)
        q_logp = -compute_Muller_potential(alpha, torch.from_numpy(samples)).numpy()

        Fq = -np.log(alpha_0*Z_0 + alpha_1*Z_1)
        Fp_DKL_pq = -np.mean(p_logp - p_logq) + Fq
        Fp_DKL_qp = np.mean(q_logq - q_logp) + Fq

        energy_matrix = (-1)*np.stack([np.concatenate([q_logq, p_logq]),
                                       np.concatenate([q_logp, p_logp])],
                                      0)
        num_conf = np.array([q_logq.shape[0], p_logq.shape[0]])
        fastmbar = FastMBAR(energy_matrix, num_conf, verbose = False)
        Fp_BAR = fastmbar.F[-1] + Fq

        Fp_DKL_pq_two.append(Fp_DKL_pq)
        Fp_DKL_qp_two.append(Fp_DKL_qp)
        Fp_BAR_two.append(Fp_BAR)

    Fp_DKL_pq_two_repeats.append(Fp_DKL_pq_two)
    Fp_DKL_qp_two_repeats.append(Fp_DKL_qp_two)
    Fp_BAR_two_repeats.append(Fp_BAR_two)


Fp_DKL_pq_two_repeats = np.array(Fp_DKL_pq_two_repeats)
Fp_DKL_qp_two_repeats = np.array(Fp_DKL_qp_two_repeats)
Fp_BAR_two_repeats = np.array(Fp_BAR_two_repeats)

fig = plt.figure(1)
fig.clf()
plt.contourf(U, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
idx = np.random.choice(range(samples.shape[0]), size = 1000)
plt.plot(samples[:,0][idx], samples[:,1][idx], '.')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.colorbar()
plots_samples.savefig()
plots_samples.close()

fig = plt.figure(2)
fig.clf()
num_steps = Fp_DKL_pq_two_repeats.shape[0]

p1, = plt.plot(range(num_steps), np.repeat([np.mean(Fp_DKL_pq_one_repeats)], num_steps), '-s', color = 'orange', markerfacecolor = 'none')
p2, = plt.plot(range(num_steps), np.mean(Fp_DKL_pq_two_repeats, -1), '-o', color = 'orange', markerfacecolor = 'none')

p3, = plt.plot(range(num_steps), np.repeat([np.mean(Fp_DKL_qp_one_repeats)], num_steps), '-s', color = 'green', markerfacecolor = 'none')
p4, = plt.plot(range(num_steps), np.mean(Fp_DKL_qp_two_repeats, -1), '-o', color = 'green', markerfacecolor = 'none')

p5, = plt.plot(range(num_steps), np.repeat([np.mean(Fp_BAR_one_repeats)], num_steps), '-s', color = 'blue', markerfacecolor = 'none')
p6, = plt.plot(range(num_steps), np.mean(Fp_BAR_two_repeats, -1), '-o', color = 'blue', markerfacecolor = 'none')

# p5, = plt.plot(range(num_steps), np.repeat([np.mean(Fp_BAR_one_repeats)], num_steps), '-<', color = 'blue')
# p6, = plt.plot(range(num_steps), np.mean(Fp_BAR_two_repeats, -1), '->', color = 'blue')

p7 = plt.axhline(y = Fp, color = "k")

plt.legend([(p7), (p5, p6), (p1, p2), (p3, p4)], ["Exact", "BAR", r"$\leftangle -W_{A \rightarrow A^{\circ}} \rightangle$", r"$\leftangle W_{A^{\circ} \rightarrow A} \rightangle$"],
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, ncol = 2)
plt.ylim([-7,-1.5])
plt.xlabel("Number of EM steps")
plt.tight_layout()
plt.savefig("./output/mixture_of_normal/F.eps")

exit()

    
exit()

plt.errorbar(range(num_steps), np.mean(Fp_DKL_pq_two_repeats, -1),
             yerr = np.std(Fp_DKL_pq_two_repeats, -1), label = "DKL_pq_two")
plt.errorbar(range(num_steps), np.mean(Fp_DKL_qp_two_repeats, -1),
             yerr = np.std(Fp_DKL_qp_two_repeats, -1), label = "DKL_qp_two")
plt.errorbar(range(num_steps), np.mean(Fp_BAR_two_repeats, -1),
             yerr = np.std(Fp_BAR_two_repeats, -1), label = "BAR_two")

plt.errorbar(range(num_steps), np.repeat([np.mean(Fp_DKL_pq_one_repeats)], num_steps),
             yerr = np.repeat(np.std(Fp_DKL_pq_one_repeats), num_steps), label = "DKL_pq_one")
plt.errorbar(range(num_steps), np.repeat([np.mean(Fp_DKL_qp_one_repeats)], num_steps),
             yerr = np.repeat(np.std(Fp_DKL_qp_one_repeats), num_steps), label = "DKL_qp_one")
plt.errorbar(range(num_steps), np.repeat([np.mean(Fp_BAR_one_repeats)], num_steps),
             yerr = np.repeat(np.std(Fp_BAR_one_repeats), num_steps), label = "BAR_one")


#### normal distribution with three components ####
#################################################
x_mean_0 = np.copy(x_mean)
x_mean_1 = np.copy(x_mean)
x_mean_2 = np.copy(x_mean)

x_cov_0 = np.copy(x_cov)
x_cov_1 = np.copy(x_cov)
x_cov_2 = np.copy(x_cov)

x_mean_0 = np.array([-0.5, 1.5])
x_mean_1 = np.array([0.0, 0.5])
x_mean_2 = np.array([0.5, 0.0])

# x_mean_0 = np.array([0.0, 0.0])
# x_mean_1 = np.array([0.0, 0.3])
# x_mean_2 = np.array([0.1, 0.0])

x_cov_0 = np.array([[1.0, 0.0],
                    [0.0, 1.0]])
x_cov_1 = np.array([[1.0, 0.0],
                    [0.0, 1.0]])
x_cov_2 = np.array([[1.0, 0.0],
                    [0.0, 1.0]])



alpha_0 = 0.33
alpha_1 = 0.33
alpha_2 = 1 - alpha_0 - alpha_1

## EM algorithm
curren_log_prob = -np.inf
stop_criteria = 1e-6
idx_step = 0
parameters = [{'x_mean_0': x_mean_0, 'x_mean_1': x_mean_1, 'x_mean_2': x_mean_2,
               'x_cov_0': x_cov_0, 'x_cov_1': x_cov_1, 'x_cov_2': x_cov_2,
               'alpha_0': alpha_0, 'alpha_1': alpha_1, 'alpha_2': alpha_2}]
while True:
    ## EM algorithm
    dist_0 = multivariate_normal(x_mean_0, x_cov_0)
    dist_1 = multivariate_normal(x_mean_1, x_cov_1)
    dist_2 = multivariate_normal(x_mean_2, x_cov_2)
    
    Z_0, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_0],
                               epsabs = 1e-12, epsrel = 1e-12)
    Z_1, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_1],
                               epsabs = 1e-12, epsrel = 1e-12)
    Z_2, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_2],
                               epsabs = 1e-12, epsrel = 1e-12)

    print("Z_0: {:.2f}, Z_1: {:.2f}, Z_2: {:.2f}, alpha_0: {:.3f}, alpha_1: {:.3f}, alpha_2: {:.5f}".format(Z_0, Z_1, Z_2, alpha_0, alpha_1, alpha_2))
    print("x_mean_0: ", x_mean_0)    
    print("x_mean_1: ", x_mean_1)
    print("x_mean_2: ", x_mean_2)
    
    logp_0 = dist_0.logpdf(x) - np.log(Z_0) + np.log(alpha_0)
    logp_1 = dist_1.logpdf(x) - np.log(Z_1) + np.log(alpha_1)
    logp_2 = dist_2.logpdf(x) - np.log(Z_2) + np.log(alpha_2)    

    p_0 = np.exp(logp_0)
    p_1 = np.exp(logp_1)
    p_2 = np.exp(logp_2)    

    w_0 = p_0 / (p_0 + p_1 + p_2)
    w_1 = p_1 / (p_0 + p_1 + p_2)
    w_2 = 1 - w_0 - w_1

    x_mean_0 = np.sum(x * w_0[:,np.newaxis], 0)/np.sum(w_0)
    x_mean_1 = np.sum(x * w_1[:,np.newaxis], 0)/np.sum(w_1)
    x_mean_2 = np.sum(x * w_2[:,np.newaxis], 0)/np.sum(w_2)    

    x_cov_0 = np.matmul(((x - x_mean_0)*w_0[:, np.newaxis]).T, (x - x_mean_0)*w_0[:, np.newaxis])/np.sum(w_0)
    x_cov_1 = np.matmul(((x - x_mean_1)*w_1[:, np.newaxis]).T, (x - x_mean_1)*w_1[:, np.newaxis])/np.sum(w_1)
    x_cov_2 = np.matmul(((x - x_mean_2)*w_2[:, np.newaxis]).T, (x - x_mean_2)*w_2[:, np.newaxis])/np.sum(w_2)    

    alpha_0 = np.sum(w_0)/(np.sum(w_0) + np.sum(w_1) + np.sum(w_2))
    alpha_1 = np.sum(w_1)/(np.sum(w_0) + np.sum(w_1) + np.sum(w_2))
    alpha_2 = 1 - alpha_0 - alpha_1

    log_prob = np.mean(np.log(p_0 + p_1 + p_2))
    print("idx_step: {}, log_prob: {:2f}".format(idx_step, log_prob))

    parameters.append({'x_mean_0': x_mean_0, 'x_mean_1': x_mean_1, 'x_mean_2': x_mean_2,
                       'x_cov_0': x_cov_0, 'x_cov_1': x_cov_1, 'x_cov_2': x_cov_2,
                       'alpha_0': alpha_0, 'alpha_1': alpha_1, 'alpha_2': alpha_2})
    
    if log_prob - curren_log_prob < stop_criteria and idx_step > 10:
        break
    else:
        idx_step += 1
        curren_log_prob = log_prob

Fp_DKL_pq_list = []
Fp_DKL_qp_list = []
Fp_BAR_list = []
for idx_step in range(len(parameters)):
    print(idx_step)    
    x_mean_0 = parameters[idx_step]['x_mean_0']
    x_mean_1 = parameters[idx_step]['x_mean_1']
    x_mean_2 = parameters[idx_step]['x_mean_2']    

    x_cov_0 = parameters[idx_step]['x_cov_0']
    x_cov_1 = parameters[idx_step]['x_cov_1']
    x_cov_2 = parameters[idx_step]['x_cov_2']    

    alpha_0 = parameters[idx_step]['alpha_0']
    alpha_1 = parameters[idx_step]['alpha_1']    
    alpha_2 = parameters[idx_step]['alpha_2']
    
    dist_0 = multivariate_normal(x_mean_0, x_cov_0)
    dist_1 = multivariate_normal(x_mean_1, x_cov_1)
    dist_2 = multivariate_normal(x_mean_2, x_cov_2)    
    
    Z_0, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_0],
                               epsabs = 1e-12, epsrel = 1e-12)
    Z_1, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_1],
                               epsabs = 1e-12, epsrel = 1e-12)
    Z_2, _ = integrate.dblquad(compute_density, x1_min, x1_max, x2_min, x2_max, [dist_2],
                               epsabs = 1e-12, epsrel = 1e-12)
    
    logp_0 = dist_0.logpdf(x) - np.log(Z_0) + np.log(alpha_0)
    logp_1 = dist_1.logpdf(x) - np.log(Z_1) + np.log(alpha_1)
    logp_2 = dist_2.logpdf(x) - np.log(Z_2) + np.log(alpha_2)    

    p_0 = np.exp(logp_0)
    p_1 = np.exp(logp_1)
    p_2 = np.exp(logp_2)    
    
    p_logq = np.log(p_0 + p_1 + p_2)
    p_logp = -compute_Muller_potential(alpha, torch.from_numpy(x)).numpy()
    
    num_samples = x.shape[0]
    samples_0 = dist_0.rvs(num_samples)
    samples_1 = dist_1.rvs(num_samples)
    samples_2 = dist_1.rvs(num_samples)        
    
    flag = (samples_0[:,0] >= x1_min) & (samples_0[:,0] <= x1_max) &\
           (samples_0[:,1] >= x2_min) & (samples_0[:,1] <= x2_max)  
    samples_0 = samples_0[flag]

    flag = (samples_1[:,0] >= x1_min) & (samples_1[:,0] <= x1_max) &\
           (samples_1[:,1] >= x2_min) & (samples_1[:,1] <= x2_max)  
    samples_1 = samples_1[flag]

    flag = (samples_2[:,0] >= x1_min) & (samples_2[:,0] <= x1_max) &\
           (samples_2[:,1] >= x2_min) & (samples_2[:,1] <= x2_max)  
    samples_2 = samples_2[flag]
    
    num_samples_0 = int(num_samples*alpha_0)
    num_samples_1 = int(num_samples*alpha_1)
    num_samples_2 = int(num_samples*alpha_2)    

    samples_0 = samples_0[0:num_samples_0]
    samples_1 = samples_1[0:num_samples_1]
    samples_2 = samples_2[0:num_samples_2]        
    
    samples = np.concatenate([samples_0, samples_1, samples_2])
    logp_0 = dist_0.logpdf(samples) - np.log(Z_0) + np.log(alpha_0)
    logp_1 = dist_1.logpdf(samples) - np.log(Z_1) + np.log(alpha_1)
    logp_2 = dist_2.logpdf(samples) - np.log(Z_2) + np.log(alpha_2)    

    p_0 = np.exp(logp_0)
    p_1 = np.exp(logp_1)
    p_2 = np.exp(logp_2)    
    
    q_logq = np.log(p_0 + p_1 + p_2)
    q_logp = -compute_Muller_potential(alpha, torch.from_numpy(samples)).numpy()

    Fq = -np.log(alpha_0*Z_0 + alpha_1*Z_1 + alpha_2*Z_2)
    Fp_DKL_pq = -np.mean(p_logp - p_logq) + Fq
    Fp_DKL_qp = np.mean(q_logq - q_logp) + Fq

    energy_matrix = (-1)*np.stack([np.concatenate([q_logq, p_logq]),
                                   np.concatenate([q_logp, p_logp])],
                                  0)
    num_conf = np.array([q_logq.shape[0], p_logq.shape[0]])
    fastmbar = FastMBAR(energy_matrix, num_conf, verbose = False)
    Fp_BAR = fastmbar.F[-1] + Fq

    Fp_DKL_pq_list.append(Fp_DKL_pq)
    Fp_DKL_qp_list.append(Fp_DKL_qp)
    Fp_BAR_list.append(Fp_BAR)


exit()






