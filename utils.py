#!/usr/bin/env python3

from enum import Enum
from copy import deepcopy
import re


import datetime
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from sklearn.cluster import AgglomerativeClustering as ACl

from typing import Tuple

from tqdm import tqdm

import lap

def print_params(func,*args,**kwargs):
    def wrapper(*args,**kwargs):
        print("Function : {} | Parameters : {} {} ".format(func.__name__,
                                                           args,
                                                           kwargs),
             )    
        return func(*args,**kwargs)
    return wrapper


def read_file(pth : str,
              index_col = 0,
              )->pd.DataFrame:
    df = pd.read_csv(pth,
                     sep = '\t',
                     header = 0,
                     index_col = index_col,
                     )
    return df

def filter_genes(mat,
                 min_occur : int = 5,
                 min_expr : int = 0,
                 )->None:

    keep_genes = (np.sum(mat.values > 0, axis = 0) > min_occur).astype(int)
    keep_genes *= (np.sum(mat.values, axis = 0) > min_expr).astype(int)
    keep_genes *= np.array([not bool(re.match('^RP|^MT',x.upper())) for x in mat.columns]).astype(int)
    mat = mat.iloc[:,keep_genes.astype(bool)]
    return mat

def propagate(cd : CountData,
              thrs : float = 10e-10,
              dt : float = 0.01,
              stopafter : int = 10e10,
              normalize : bool = True,
              diffusion_rate : int = 1,
              shuffle : bool = False,
              )-> np.ndarray:

    D = diffusion_rate
    n_genes =  cd.G 
    times = np.zeros(n_genes)
    n_saturated = cd.saturated.shape[0]

    if n_saturated < 1:
        print("[ERROR] : No Saturated spots")
        sys.exit(-1)
    else:
        print("[INFO] : {} Saturated Spots".format(n_saturated))

    if normalize:
        rowsums = np.sum(cd.cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cd.cnt.values,rowsums,where = rowsums > 0)
    else:
        ncnt = cd.cnt.values

    if shuffle:
        shf = np.random.permutation(ncnt.shape[0])
        ncnt = ncnt[shf,:]
        iterable = range(n_genes)
    else:
        iterable = tqdm(range(n_genes))
    
    # Get neighbour indices
    nidx = cd.get_allnbr_idx(cd.saturated)

    # Propagate in time
    for gene in iterable:

        conc = ncnt[:,gene].astype(float)
        q_95 = np.quantile(conc,0.95,interpolation = 'nearest')
        conc[conc > q_95] = q_95
        maxDelta = np.inf

        time = 0
        while maxDelta > thrs and conc[cd.saturated].sum() > 0:
            if time / dt > stopafter:
                genename = cd.cnt.columns[gene]
                print("WARNING : Gene : {} did not convege".format(genename))
                break
            time +=dt

            d2 = cd.laplacian(conc[cd.saturated],
                              conc[nidx],
                              cd.h[cd.saturated])
            dcdt = D*d2
            conc[cd.saturated] = conc[cd.saturated] +  dcdt*dt 

            conc[conc < 0] = 0
            times[gene] = time
            maxDelta = np.max(np.abs(dcdt*dt)) / dcdt.shape[0]

    return times

def clean_axes(ax : plt.Axes,
                )->None:

        ax.set_aspect('equal')
        ax.set_yticks([])
        ax.set_xticks([])
        for pos in ax.spines.keys():
            ax.spines[pos].set_visible(False)

def visualize_genes(cnt : CountData,
                    crd : np.ndarray,
                    times : np.ndarray,
                    n_genes : int = 20,
                    ncols : int = 5,
                    side_size : float = 3,
                    qscale : float = None ,
                    normalize : bool = True,
                    pltargs : dict = None,
                    ) -> Tuple:


    if normalize:
        rowsums = np.sum(cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cnt.values,rowsums,where = rowsums > 0)
    else:
        ncnt = cnt.values

    topgenes = np.argsort(times)[::-1][0:n_genes]
    nrows = np.ceil(n_genes / ncols).astype(int)

    figsize = (1.2 * ncols * side_size,
               1.2 * nrows * side_size)

    fig,ax = plt.subplots(nrows,
                          ncols,
                          figsize=figsize)
    ax = ax.flatten()

    _pltargs = {'s':40,
                'edgecolor':'black',
                'cmap':plt.cm.PuRd,
                }

    if pltargs is not None:
        for k,v in pltargs.items():
            _pltargs[k] = v
            if k == 'cmap' and isinstance(k,str):
                _pltargs[k] = eval(v)

    for ii in range(n_genes):
        vals = ncnt[:,topgenes[ii]].reshape(-1,)
        if qscale is not None:
            if qscale > 0 and qscale < 1:
                vals_q = np.quantile(vals,qscale)
                vals[vals > vals_q] = vals_q
            else:
                print('WARNING : {} is not a proper quantile value'.format(qscale),
                      'within range (0,1)')

        ax[ii].set_title('Gene : {} \nPotential : {:0.3f}'.format(cnt.columns[topgenes[ii]],
                                                             times[topgenes[ii]]))
        ax[ii].scatter(crd[:,0],
                       crd[:,1],
                       c = vals,
                       **_pltargs,
                      )
        clean_axes(ax[ii])

    return (fig,ax)


def get_eigenpatterns( mat : np.ndarray,
                       thrs : float = 0.99,
                       normalize : bool = True,
                       )-> np.ndarray :


    U,S,_ = np.linalg.svd(mat - mat.mean(axis = 1).reshape(-1,1))
    nonzero = np.where(S > 0)[0]
    evecs = U[:,nonzero]
    evals = S**2 / mat.shape[0]

    ncomps = np.cumsum(evals)
    ncomps = np.argmax(ncomps / ncomps[-1] > thrs)

    return evecs[:,0:ncomps]

def get_eigenscores(mat,
                    evecs,
                    )-> np.ndarray:

    coef = np.linalg.lstsq(evecs,
                           mat,
                           rcond = None)[0]

    coef_norms = np.linalg.norm(coef,axis = 0)
    normed_coefs = coef / coef_norms.reshape(1,-1)

    return normed_coefs


def get_eigen_dmat(eigenscores : np.ndarray,
                   normalized : bool = True,
                      ) -> np.ndarray :


    n_pats = eigenscores.shape[1]
    dmat = np.zeros((n_pats,n_pats))

    if not normalized:
        nrm = np.linalg.norm(eigenscores,axis = 0)
        eigenscores = eigenscores / nrm.reshape(1,-1)

    for ii in range(n_pats-1):
        u = eigenscores[:,ii]
        for jj in range(ii+1,n_pats):
            v = eigenscores[:,jj]
            dmat[ii,jj] = np.arccos(np.dot(u,v))
            dmat[jj,ii] = dmat[ii,jj]

    return dmat

def cluster_patterns(dmat : np.ndarray,
                     n_patterns : int,
                     )-> np.ndarray:


    if n_patterns > dmat.shape[0]:
        string = ' '.join(["Number of patterns larger",
                           "than number of samples"])
        raise Exception(string)

    cluster = ACl(n_clusters = n_patterns,
                  affinity = 'precomputed',
                  linkage = 'complete')

    cidx = cluster.fit_predict(dmat)

    return cidx

def visualize_clusters(cd : CountData,
                       times : np.ndarray,
                       ntopgenes : np.ndarray,
                       ncols : int,
                       threshold : float = 0.8,
                       side_size : float = 3,
                       pltargs : dict = None,
                       normalize : bool = True,
                       show_genes : int = None,
                       ):
    
    _pltargs = {'s':40,
                'edgecolor':'black',
                'cmap':plt.cm.PuRd,
                }

    if pltargs is not None:
        for k,v in pltargs.items():
            _pltargs[k] = v
            if k == 'cmap' and isinstance(k,str):
                _pltargs[k] = eval(v)

    toppatterns = np.argsort(times)[::-1][0:ntopgenes]
    genes = cd.cnt.columns.values[toppatterns]
    epats = get_eigenpatterns(cd.cnt.values[:,toppatterns],thrs = threshold)
    n_patterns = epats.shape[1]
    
    nrows = np.ceil(n_patterns / ncols).astype(int)
    figsize = (1.2 * ncols * side_size,
               1.2 * nrows * side_size)

    eigviz = list(plt.subplots(nrows,ncols,figsize = figsize))
    eigviz[0].suptitle('Eigenpatterns')
    eigviz[-1] = eigviz[-1].flatten()

    for k in range(n_patterns):
        eigviz[-1][k].scatter(cd.crd[:,0],cd.crd[:,1],
                              c = epats[:,k],
                              **_pltargs,
                              )

        eigviz[-1][k].set_aspect('equal')
        clean_axes(eigviz[-1][k])
        
    if normalize:
        rowsums = np.sum(cd.cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cd.cnt.values,rowsums,where = rowsums > 0)
    else:
        ncnt = cd.cnt.values


    escores = get_eigenscores(ncnt[:,toppatterns],epats)
    labels = cluster_patterns(get_eigen_dmat(escores),
                              n_patterns = n_patterns)

    vizlist = []
    uni_labels = np.unique(labels)

    for k,lab in enumerate(uni_labels):
        pos = np.where(labels == lab)[0]

        if show_genes is not None:
            pos = pos[0:np.min((show_genes,pos.shape[0])).astype(int)]

        nrows = np.ceil(pos.shape[0] / ncols).astype(int)

        figsize = (1.2 * ncols * side_size,
                   1.2 * nrows * side_size)

        vizlist.append(list(plt.subplots(nrows,
                                         ncols,
                                         figsize = figsize)))

        vizlist[-1][0].suptitle("Eigengroup {}".format(k))

        vizlist[-1][1] = vizlist[-1][1].flatten()

        for ii in range(pos.shape[0]):
            vizlist[-1][1][ii].set_title("Gene : {}".format(genes[pos[ii]]))
            vizlist[-1][1][ii].scatter(cd.crd[:,0],
                                       cd.crd[:,1],
                                       c = cd.cnt.values[:,toppatterns[pos[ii]]],
                                       **_pltargs,
                                       )

            vizlist[-1][1][ii].set_aspect('equal')
            clean_axes(vizlist[-1][1][ii])

        for jj in range(ii+1,ncols*nrows):
            clean_axes(vizlist[-1][1][jj])
            vizlist[-1][1][jj].set_visible(False)

    return (eigviz,vizlist)


def change_crd_index(df : pd.Index,
                     new_crd : np.ndarray) -> pd.Index :

    new_idx = [ 'x'.join(crd[x,:].astype(str)) for\
                x in range(crd.shape[0])]

    new_idx = pd.Index(new_idx)

    old_idx  = df.index
    df.index = new_idx

    return (df,old_idx) 

def timestamp() -> str:
    return re.sub(':|-|\.| |','',str(datetime.datetime.now()))

