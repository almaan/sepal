#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

np.random.seed(1337)

pth = "/home/alma/w-projects/spatential/data/fake-patterns/20200127161150201659-pattern-count-matrix.tsv"
odir = "/home/alma/w-projects/spatential/data/fake-patterns/ablation-sets"

cnt = pd.read_csv(pth, sep = '\t',header = 0,index_col = 0)
crd = np.array([x.split('x') for x in cnt.index.values]).astype(float)

n_shuffle = np.array([0,100,500,900])

for ii in range(10):
    vals = cnt.values[:,ii*10]
    n_spots = crd.shape[0]


    abl = pd.DataFrame(np.zeros((crd.shape[0],n_shuffle.shape[0])),
                    index = cnt.index,
                    columns = ["Shuffled : " + str(x) for x in n_shuffle],
                    )
    all_idx = np.arange(n_spots)
    # fig,ax  = plt.subplots(4,5)
    # ax = ax.flatten()

    for k,shuf in enumerate(n_shuffle):
        expr = vals.copy()
        idx = np.random.choice(all_idx,size = shuf,replace=False)
        expr[idx] = expr[np.random.permutation(idx)]
        abl.iloc[:,k] = expr
        # ax[k].scatter(crd[:,0],crd[:,1],cmap = plt.cm.jet,c = expr,s = 20)
        # ax[k].set_aspect("equal")
        # ax[k].set_title(abl.columns.values[k])


    abl.to_csv(osp.join(odir,"ablated-" + str(ii) + ".tsv"),sep = '\t',header = True, index = True)

plt.close("all")
