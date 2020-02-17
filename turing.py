#!/usr/bin/env python3

import os
import os.path as osp

from typing import Union, List, Tuple

import re
import datetime

import numpy as np
import pandas as pd

import PIL.Image as im

import argparse as arp

import matplotlib.pyplot as plt


def lap(w,h = 1):
    top = w[0:-2,1:-1]
    left = w[1:-1,0:-2]
    bottom = w[2:,1:-1]
    right = w[1:-1,2:]
    center = w[1:-1,1:-1]

    return (top + left + bottom + right - 4.0 * center) / h**2

def du(uu,vv, a = 2.1,Du = 1.0, Dv = 1.0, delta = 1.0,beta = 1.0):
    u = uu[1:-1,1:-1]
    v = vv[1:-1,1:-1]
    return u*(1-u) - u*v / (u+a) + Du * lap(uu) 

def dv(uu,vv, a = 2.1,Du = 1.0, Dv = 1.0, delta = 1.0,beta = 1.0):
    u = uu[1:-1,1:-1]
    v = vv[1:-1,1:-1]
    return v*delta*(1-beta*v/u) + Dv*lap(vv)

def make_pattern(x_dim,y_dim):

    U = np.random.random((y_dim,x_dim))
    V = np.random.random((y_dim,x_dim))

    n_steps = 1000
    dt = 0.01

    prms1 = dict(a = 0.1,
                 Du = 0.1,
                 Dv = 7,
                 beta = 0.25,
                 delta = 0.2,
                 )

    prms2 = dict(a = 0.1,
                Du = 0.1,
                Dv = 4,
                beta = 0.25,
                delta = 0.2,
                )

    prms3 = dict(a = 0.1,
                    Du = 0.1,
                    Dv = 5,
                    beta = 0.15,
                    delta = 0.8,
                    )

    prms4 = dict(a = 0.1,
                    Du = 0.1,
                    Dv = 4,
                    beta = 0.15,
                    delta = 0.3,
                    )




    for i in range(n_steps):

        U[1:-1,1:-1] += du(U,V,**prms1)*dt
        V[1:-1,1:-1] += dv(U,V,**prms1)*dt

        for Z in (U,V):
            Z[0, :] = Z[1, :]
            Z[-1, :] = Z[-2, :]
            Z[:, 0] = Z[:, 1]
            Z[:, -1] = Z[:, -2]

    return U.flatten()



def make_count_matrix(n_patterns : int = 10,
                      n_friends : int = 3,
                      mult_factors : Union[List,Tuple,np.ndarray] = [0.5,1,2],
                      x_dim : int = 30,
                      y_dim : int = 30,
                      )->pd.DataFrame:

    x = np.arange(x_dim)
    y = np.arange(y_dim)

    x,y = np.meshgrid(x,y)
    x = x.flatten()
    y = y.flatten()
    
    n_genes = n_patterns * (1 + n_friends*len(mult_factors))
    g_iter = iter(range(n_genes))
    n_spots = x_dim*y_dim
    gene_names = list()

    gmat = np.zeros((n_spots,n_genes),dtype = np.float)

    for num in range(n_patterns):
        gene_vals = make_pattern(x_dim = x_dim,y_dim = y_dim)
        gene_vals *= 100
        gmat[:,next(g_iter)] = gene_vals

        gene_names.append("P" + str(num) + "-" + "O")

        for friend in range(n_friends):
            for mult in mult_factors:
                gmat[:,next(g_iter)] = (np.random.permutation(gene_vals) * mult)
                gene_names.append("P" + str( num ) + \
                                "-F" + str(friend) + "-T" + str( mult ))

    spot_names = [str( xx ) + 'x' + \
                  str( yy ) for \
                  xx,yy in zip(x,y)]

    gmat = pd.DataFrame(gmat,
                        index = spot_names,
                        columns = gene_names)

    return gmat


matrix = make_count_matrix()

crd = np.array( [x.split('x') for x in matrix.index] ).astype(float)

plt.scatter(crd[:,0],crd[:,1],c = matrix.values[:,10],s = 80)
plt.show()

matrix.to_csv("/tmp/turing-test-5.tsv",sep = '\t',header =True, index = True)
