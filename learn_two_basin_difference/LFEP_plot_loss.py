import sys
sys.path.append('../')
from RealNVP import *
from functions import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

beta = 0.05
num_steps = list(map(int, np.logspace(1.4,4.3,50)))
basin_id = 2

## extract training and validation loss
with open('../output/LFEP/loss_record_basin%d_beta%.3f.pkl'%(basin_id, beta), 'rb') as file_handle:
	data = pickle.load(file_handle)
loss_train_record, loss_valid_record = data['training_loss'], data['validation_loss']

loss_train_record = [loss_train_record[i] for i in num_steps]
## plot
# training
fig = plt.figure(0)
fig.clf()
plt.plot(num_steps, loss_train_record)
plt.xscale("log")
plt.xlabel(r"steps")
plt.ylabel(r"training loss")
plt.tight_layout()
plt.savefig('../output/LFEP/loss_train_record_basin%d_beta%.3f.pdf'%(basin_id, beta))

# validation
fig = plt.figure(0)
fig.clf()
plt.plot(num_steps,loss_valid_record)
plt.xscale("log")
plt.xlabel(r"steps")
plt.ylabel(r"validation loss")
plt.tight_layout()
plt.savefig('../output/LFEP/loss_valid_record_basin%d_beta%.3f.pdf'%(basin_id, beta))


sys.exit()
