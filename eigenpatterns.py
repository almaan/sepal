#!/usr/bin/env python3


import numpy as np
import pandas as pd
from PIL import Image

from sklearn.cluster import AgglomerativeClustering as ACl

cnt_pth = "/home/alma/Documents/PhD/papers/STSC/data/bc/st/under_tissue/ut_H1_stdata_filtered.tsv.gz"
pat_pth = "/tmp/diffusion-times.tsv"

cnt = pd.read_csv(cnt_pth,sep='\t',header = 0, index_col = 0)
pat = pd.read_csv(pat_pth,sep='\t',header = 0, index_col = 0)

pat = pat.iloc[np.argsort(pat.values.flatten())[::-1],:]
spotsums = cnt.values.sum(axis = 1).reshape(-1,1)
top = pat.index[0:50]

selcnt = cnt.loc[:,top]
mat = selcnt.values / spotsums

cmat = np.cov((mat - mat.mean(axis = 1).reshape(-1,1)))

U,S,_ = np.linalg.svd(mat - mat.mean(axis = 1).reshape(-1,1))
nonzero = np.where(S > 0)[0]
evecs = U[:,nonzero]
evals = S**0.5 / mat.shape[0]


evals,evecs = np.linalg.eig(cmat)

evals = np.real(evals)
evecs = np.real(evecs)

crd = np.array([x.lstrip('X').split('x') for x in cnt.index.values ]).astype(np.float)

ngenes = 15
ncols = 5
nrows = np.ceil(ngenes / ncols).astype(int)

fig, ax = plt.subplots(nrows,ncols)
ax = ax.flatten()

for k in range(ngenes):
    ax[k].scatter(crd[:,0],crd[:,1],
              c = evecs[:,k],
              cmap = plt.cm.PuRd,
              s = 10)
    ax[k].set_aspect('equal')

plt.show()

ncomps = 10 
test = np.linalg.lstsq(evecs[:,0:ncomps],mat,rcond = None)[0]
norms = np.linalg.norm(test,axis = 0)
normed_vectors = test / norms.reshape(1,-1)

dmat = np.zeros((50,50))
for ii in range(50-1):
    u = normed_vectors[:,ii]
    for jj in range(ii+1,50):
        v = normed_vectors[:,jj]
        dmat[ii,jj] = np.arccos(np.dot(u,v))
        dmat[jj,ii] = dmat[ii,jj]

cluster = ACl(n_clusters = ncomps,
              affinity = 'precomputed',
              linkage = 'complete')

labels = cluster.fit_predict(dmat)

uni_labels = np.unique(labels)
ncols = 5

for k,lab in enumerate(uni_labels):
    pos = np.where(labels == lab)[0]
    nrows = np.ceil((labels == lab).sum() / ncols).astype(int)
    _,ax = plt.subplots(nrows,ncols)
    ax = ax.flatten()
    for ii in range((labels == lab).sum()):
       ax[ii].scatter(crd[:,0],
                      crd[:,1],
                      c = mat[:,pos[ii]],
                      cmap = plt.cm.PuRd,
                      s = 5)

       ax[ii].set_aspect('equal')

    for jj in range(ii+1,ncols*nrows):
        ax[jj].set_visible(False)

    plt.show()
