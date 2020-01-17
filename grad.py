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
pth = "/home/alma/Documents/PhD/papers/STSC/data/kidney/st/raw/counts_under_tissue_hgnc/EXP2_NHK3_C1.stdata.tsv"

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

times = utils.propagate(cd,)

fig,ax = utils.visualize_genes(cd, times,qscale = 0.99)
plt.show()
