#!/usr/bin/env python3

import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import re


def laplacian(centers : np.ndarray,
              nbrs : np.ndarray,
              h : float,
              )-> np.ndarray:

    d2f = nbrs.sum(axis = 1) - 4*centers
    d2f = d2f / h**2

    return d2f

# Load and process data
pth = "/home/alma/Documents/PhD/papers/STSC/data/molb/st/science/Rep1_MOB.st_mat.processed.tsv"

mat = utils.read_file(pth)
keep_genes = np.sum(mat.values > 0, axis = 0) > 5 
mat = mat.iloc[:,keep_genes]
keep_genes = np.sum(mat.values, axis = 0) > 300
mat = mat.iloc[:,keep_genes]
keep_genes = [not bool(re.match('^RP|^MT',x.upper())) for x in mat.columns]
mat = mat.iloc[:,keep_genes]
mat.index = pd.Index([x.replace('X','') for x in mat.index])

cd = utils.CountData(mat,
                     radius = 1.0,
                     n_neighbours = 4,
                     rotate = None)

nidx = cd.get_allnbr_idx(cd.saturated)
rowsums = np.sum(cd.cnt.values,axis = 1).reshape(-1,1)
ncnt = np.divide(cd.cnt.values,rowsums,where = rowsums > 0)

#%% Diffusion study--------
ngenes = ncnt.shape[1] 
thrs = 10e-8
D = 1
dt = 0.1
step = 0
genetime = np.zeros(ngenes)
times = np.zeros(ngenes)

for gene in range(ngenes):
    conc = ncnt[:,gene].astype(float)
    err = np.inf
    time = 0
    while err > thrs:
        time +=dt
        d2 = laplacian(conc[cd.saturated],conc[nidx],1)
        dcdt = D*d2
        conc[cd.saturated] = conc[cd.saturated] +  dcdt*dt
        err = np.max(np.abs(dcdt*dt)) / dcdt.shape[0]
    times[gene] = time

ndisplay = 20
ncols = 5

nrows = np.ceil(ndisplay / ncols).astype(int)
fig,ax = plt.subplots(nrows,
                      ncols)
ax = ax.flatten()
topgenes = np.argsort(times)[::-1][0:ndisplay]

for ii in range(ndisplay):
    vals = ncnt[:,topgenes[ii]].reshape(-1,)
#    vals_q = np.quantile(vals,0.99)
#    vals[vals > vals_q] = vals_q
    ax[ii].set_title('{}'.format(cd.cnt.columns[topgenes[ii]]))
    ax[ii].scatter(cd.crd[:,0],
               cd.crd[:,1],
               c = vals,
               s = 40,
               edgecolor = 'black',
               cmap = plt.cm.PuRd,
               )
    
    ax[ii].set_aspect('equal')
    ax[ii].set_yticks([])
    ax[ii].set_xticks([])
    for pos in ax[ii].spines.keys():
        ax[ii].spines[pos].set_visible(False)

plt.show()
