import sys
sys.path.append('../')
from RealNVP import *
from functions import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

beta = 0.05
num_steps = list(map(int, np.logspace(1.4,4.3, 50)))

## extract training and validation loss
with open('../output/LBAR/loss_record_beta%.3f.pkl'%beta, 'rb') as file_handle:
	data = pickle.load(file_handle)
loss_train_record, loss_valid_record = data['training_loss'], data['validation_loss']

## plot
# training
fig = plt.figure(0)
fig.clf()
plt.plot(loss_train_record)
# plt.xscale("log")
plt.xlabel(r"$steps$")
plt.ylabel(r"$loss$")
plt.tight_layout()
plt.savefig('../output/LBAR/loss_train_record_beta%.3f.pdf'%beta)

# validation
fig = plt.figure(0)
fig.clf()
plt.plot(num_steps,loss_valid_record)
# plt.xscale("log")
plt.xlabel(r"$steps$")
plt.ylabel(r"$loss$")
plt.tight_layout()
plt.savefig('../output/LBAR/loss_valid_record_beta%.3f.pdf'%beta)


sys.exit()
