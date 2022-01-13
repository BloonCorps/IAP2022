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
import sys

'''
The Muller potential functions, U(x), is defined.
The corresponding probability densities are defined as \log P(x) \propto \exp(-U(x))

'''


if __name__ == "__main__":

    ## given data
    xmax = 20
    nstep = 100
    xg = np.linspace(0,xmax,nstep)
    yg = np.cos(xg**2./8.)+1

    xx = np.linspace(0,xmax,1000)
    yy = np.cos(xx**2./8.)+1

    ## linear interpolate
    # f_interp = interp1d(xg, yg)
    # xw = xx
    # yw = f_interp(xw)

    # # plotting
    # plt.plot(xg,yg, 'o', label='given data')
    # plt.plot(xw, yw, ':', label='linear interpolant')
    # plt.plot(xx, yy, '-', label='underlying func')
    # plt.ylim([-0.1,3])
    # plt.show()
    # sys.exit()


    ## B-spline
    # degree of spline
    n = len(xg)
    k = 3
    len_knots = n+k+1

    # knots
    xmin, xmax = np.min(xg), np.max(xg)
    knots = np.linspace(xmin, xmax, len_knots-2*k)
    knots = np.concatenate( (np.array([xmin]*k),knots,np.array([xmax]*k)) )
    
    # spline coefficients
    c = yg #np.random.rand(len(yg))
    f_interp = BSpline(knots, c, k, extrapolate=False)

    # plotting
    xw = xg
    yw = f_interp(xw)
    plt.plot(xg,yg, 'o', label='given data')
    plt.plot(xw, yw, ':', label='linear interpolant')
    plt.plot(xx, yy, '-', label='underlying func')
    plt.ylim([-0.1,3])
    plt.show()

