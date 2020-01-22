#!/usr/bin/env python3

from enum import Enum
from copy import deepcopy
import re


import datetime
import time as Time
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# from sklearn.cluster import AgglomerativeClustering as ACl
# from sklearn.cluster import SpectralClustering as SpCl
# from sklearn.mixture import BayesianGaussianMixture as DP

from sklearn.cluster import OPTICS as Cl

from scipy.stats import fisher_exact as fe

from typing import Tuple

from tqdm import tqdm

from joblib import Parallel, delayed
from multiprocessing import cpu_count

import lap
import models as m


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

    keep_genes *= np.array([not bool(re.match('^RP|^MT',x.upper())) \
                            for x in mat.columns]).astype(int)

    mat = mat.iloc[:,keep_genes.astype(bool)]
    return mat

def propagate(cd : m.CountData,
              thrs : float = 1e-7,
              dt : float = 0.1,
              stopafter : int = 10e10,
              normalize : bool = True,
              diffusion_rate : int = 1,
              num_workers : int = None,
              )-> np.ndarray:

    if num_workers is None:
        num_workers = int(cpu_count())
    else:
        num_workers = min(num_workers,
                          cpu_count())

    print("[INFO] : Using {} workers".format(num_workers))

    diff_prop = {'D':diffusion_rate,
                 'thrs':thrs,
                 'dt':dt,
                 'stopafter':stopafter}

    n_saturated = cd.saturated.shape[0]
    if n_saturated < 1:
        print("[ERROR] : No Saturated spots")
        sys.exit(-1)
    else:
        print("[INFO] : {} Saturated Spots".format(n_saturated))

    if normalize:

        ncnt = cd.cnt.values

        #used means for good res
        rowSums = np.mean(ncnt,axis = 1).reshape(-1,1)

        ncnt = np.divide(ncnt,rowSums,where = rowSums > 0)

        colMax = np.max(ncnt,axis = 0).reshape(1,-1)

        ncnt = np.divide(ncnt,
                         colMax,where = colMax > 0)

        ncnt = ncnt.astype(float)

    else:
        ncnt = cd.cnt.values.astype(float)

    # get neighbour index
    nidx = cd.get_allnbr_idx(cd.saturated)
    # Propagate in time
    iterable = tqdm(range(cd.G))

    times = Parallel(n_jobs=num_workers)(delayed(stepping)(gene,
                                                 ncnt[:,gene],
                                                 cd,
                                                 nidx,
                                                 **diff_prop) for \
                                         gene in iterable)
    times = np.array(times)

    return times


def stepping(gene : int,
             conc : np.ndarray,
             cd : m.CountData,
             nidx : np.ndarray,
             thrs : float,
             D : float,
             dt : float,
             stopafter : int,
             )->float:

        maxDelta = np.inf
        time  = 0

        # q = np.quantile(conc,0.99,interpolation = 'nearest')
        # conc[conc > q] = q

        old_maxDelta = 1
        dcdt = 0
        
        while np.abs(old_maxDelta - maxDelta ) > thrs and conc[cd.saturated].sum() > 0:
            if time / dt > stopafter:
                genename = cd.cnt.columns[gene]
                print("WARNING : Gene :"
                      "{} did not converge"
                      "".format(genename))
                break

            old_maxDelta = maxDelta 

            time +=dt

            d2 = cd.laplacian(conc[cd.saturated],
                              conc[nidx],
                              cd.h[cd.saturated])

            dcdt = D*d2 

            conc[cd.saturated] += dcdt*dt 
            conc[conc < 0] = 0

            maxDelta = np.max(np.abs(dcdt)) 

        return time


def clean_axes(ax : plt.Axes,
                )->None:

        ax.set_aspect('equal')
        ax.set_yticks([])
        ax.set_xticks([])
        for pos in ax.spines.keys():
            ax.spines[pos].set_visible(False)

def visualize_genes(cnt : m.CountData,
                    crd : np.ndarray,
                    times : np.ndarray,
                    # n_genes : int = 20,
                    ncols : int = 5,
                    side_size : float = 3,
                    qscale : float = None ,
                    normalize : bool = False,
                    pltargs : dict = None,
                    ) -> Tuple:


    if normalize:
        rowsums = np.sum(cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cnt.values,rowsums,where = rowsums > 0)
    else:
        ncnt = cnt.values

    # topgenes = np.argsort(times)[::-1][0:n_genes]
    n_genes = cnt.shape[1]
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
        vals = ncnt[:,ii].reshape(-1,)
        if qscale is not None:
            if qscale > 0 and qscale < 1:
                vals_q = np.quantile(vals,qscale)
                vals[vals > vals_q] = vals_q
            else:
                print('WARNING : {} is not a proper quantile value'.format(qscale),
                      'within range (0,1)')

        ax[ii].set_title('Gene : {} \nPotential : {:0.3f}'.format(cnt.columns[ii],
                                                                  times[ii]))
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

    return evecs[:,0:ncomps+1]

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
                     delta = 0.1,
                     )-> np.ndarray:


    if n_patterns > dmat.shape[0]:
        string = ' '.join(["Number of patterns larger",
                           "than number of samples"])
        raise Exception(string)

    
    smat = np.exp(- dmat / (2 * np.pi ** 2))

    print("[INFO] : Using {} clusters ".format(n_patterns))

    cluster = SpCl(n_clusters = n_patterns,
                   affinity = 'precomputed')


    cidx = cluster.fit_predict(smat)

    return cidx

def cluster_data(counts : np.ndarray,
                 threshold : float = 0.8,
                 ):

    epats = get_eigenpatterns(counts,thrs = threshold)

    n_patterns = epats.shape[1]

    print("[INFO] : Using {} eigenpatterns".format(n_patterns))

    escores = get_eigenscores(counts,epats)

    cidx = Cl(metric = 'cosine',
              min_samples = 3).fit_predict(escores.T)

    return cidx
 
def visualize_clusters(counts : np.ndarray,
                       genes : pd.Index,
                       crd : np.ndarray,
                       labels : np.ndarray,
                       ncols : int,
                       side_size : float = 3,
                       pltargs : dict = None,
                       normalize : bool = True,
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
   
    vizlist = []
    uni_labels = np.unique(labels)

    for k,lab in enumerate(uni_labels):
        if lab == -1:
            continue

        pos = np.where(labels == lab)[0]

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
            vizlist[-1][1][ii].scatter(crd[:,0],
                                       crd[:,1],
                                       c = counts[:,pos[ii]],
                                       **_pltargs,
                                       )

            vizlist[-1][1][ii].set_aspect('equal')
            clean_axes(vizlist[-1][1][ii])

        for jj in range(ii+1,ncols*nrows):
            clean_axes(vizlist[-1][1][jj])
            vizlist[-1][1][jj].set_visible(False)

    return vizlist


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

def toprank(cnt : pd.DataFrame,
            diff : pd.DataFrame,
            )-> None:

    ginter = diff.index.intersection(cnt.columns)
    cnt = cnt.loc[:,ginter]
    diff = diff.loc[ginter]
    cnt_s = cnt.values.sum(axis = 0)

    cont = True

    while cont:

        n_examine = input("# genes from result >> ")
        n_topexpr = input("# top expressed genes >> ")

        if n_examine.isdigit() and  n_topexpr.isdigit():
            n_examine = int(n_examine)
            n_topexpr = int(n_topexpr)
        else:
            cont = False
            break

        cnt_g = cnt.columns.values
        cnt_ordr = np.argsort(cnt_s)[::-1]
        cnt_g = cnt_g[cnt_ordr]

        diff_g = diff.index.values
        diff_ordr = np.argsort(diff.values)[::-1]

        diff_g = diff_g[diff_ordr]

        sd = set(diff_g[0:n_examine])
        sc = set(cnt_g[0:n_topexpr])

        inter = sd.intersection(sc)
        ninter = len(inter)

        cmat = np.array([[ninter,n_examine -ninter,],
                        [n_topexpr - ninter,
                        cnt.shape[0]-n_examine-n_topexpr+ninter]])

        _,pval = fe(cmat)

        print()
        print("[TOP SPATIAL PATTERNS]: {} genes".format(n_examine))
        print("[TOP EXPRESSED] : {} genes".format(n_topexpr))
        print("[INTERSECTION] : {} genes ".format(ninter))
        print("[P-VALUE] : {} ".format(pval))

