import numpy as np
import torch
torch.set_default_dtype(torch.double)
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
from matplotlib import cm
from sys import exit
import pickle
from scipy.interpolate import BSpline
from scipy.interpolate import interp1d

'''
The Muller potential functions, U(x), is defined.
The corresponding probability densities are defined as \log P(x) \propto \exp(-U(x))

'''

x1_min, x1_max = -1.5, 1.0
x2_min, x2_max = -0.5, 2.0

def compute_cubic_spline(x, extent = (x1_min, x1_max, x2_min, x2_max)) -> list:
    x1_min, x1_max, x2_min, x2_max = extent

    ## spline degree
    k = 3

    ## knots of spline function
    knots1 = np.linspace(x1_min, x1_max, 10)
    knots2 = np.linspace(x2_min, x2_max, 10)

    ## length of controls in each dimension
    n1 = len(knots1)+k-1
    n2 = len(knots2)+k-1

    ## 
    knots1 = np.concatenate( ([x1_min]*k, knots1, [x1_max]*k ))
    knots2 = np.concatenate( ([x2_min]*k, knots2, [x2_max]*k ))

    spline_lst1 = []
    for i in range(n1):
        c = np.zeros(n1)
        c[i] = 1.0
        spline_lst1.append(BSpline(knots1, c, k, extrapolate=False))

    spline_lst2 = []
    for i in range(n2):
        c = np.zeros(n2)
        c[i] = 1.0
        spline_lst2.append(BSpline(knots2, c, k, extrapolate=False))

    x1, x2 = x[:,0], x[:,1]
    y1 = np.array([spl1(x1) for spl1 in spline_lst1])
    y2 = np.array([spl2(x2) for spl2 in spline_lst2])
    y = np.matmul(y1[:,:,np.newaxis], y2[:, np.newaxis, :])
    print(y)
    y = y.reshape(x1.shape[0], -1)
    print(y)


if __name__ == "__main__":

    ## given data
    x = np.array(
        [[-1.2, -0.4],
        [-1.1, -0.3],
        [-1.0, -0.2],
        [-0.8, 0.0],
        [-0.5, 0.4],
        [-0.25, 0.8],
        [0.1, 1.0],
        [0.4, 1.2],
        [0.75, 1.5],
        [0.9, 1.9]])

    compute_cubic_spline(x)

