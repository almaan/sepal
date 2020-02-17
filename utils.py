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
from matplotlib import rcParams

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from sklearn.cluster import OPTICS as Cl
from sklearn.cluster import AgglomerativeClustering as ACl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from scipy.stats import fisher_exact as fe

from typing import Tuple,Dict

from tqdm import tqdm

from joblib import Parallel, delayed
from multiprocessing import cpu_count

import lap
import models as m

rcParams.update({'figure.max_open_warning': 0})

def print_params(func,*args,**kwargs):
    def wrapper(*args,**kwargs):
        print("Function : {} | Parameters : {} {} ".format(func.__name__,
                                                           args,
                                                           kwargs))
        return func(*args,**kwargs)
    return wrapper


def read_file(pth : str,
              index_col = 0,
              )->pd.DataFrame:
    df = pd.read_csv(pth,
                     sep = '\t',
                     header = 0,
                     engine = 'c',
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
              thrs : float = 1e-8,
              dt : float = 0.001,
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

        # rowSums = np.mean(ncnt,axis = 1).reshape(-1,1)

        # ncnt = np.divide(ncnt,rowSums,where = rowSums > 0)

        ncnt = np.log2(ncnt + 2) 

        colMax = np.max(np.abs(ncnt),axis = 0).reshape(1,-1)

        ncnt = np.divide(ncnt,
                         colMax,
                         where = colMax > 0)

        ncnt = ncnt.astype(float)

    else:
        ncnt = cd.cnt.values.astype(float)


    # get neighbour index
    snidx = cd.get_satnbr_idx(cd.saturated)
    unidx = cd.get_unsatnbr_idx(cd.unsaturated)

    # Propagate in time
    iterable = tqdm(range(cd.G))

    times = Parallel(n_jobs=num_workers)(delayed(stepping)(gene,
                                                 ncnt[:,gene],
                                                 cd,
                                                 snidx,
                                                 unidx,
                                                 **diff_prop) for \
                                         gene in iterable)
    times = np.array(times)

    return times


def stepping(gene : int,
             conc : np.ndarray,
             cd : m.CountData,
             snidx : np.ndarray,
             unidx : np.ndarray,
             thrs : float,
             D : float,
             dt : float,
             stopafter : int,
             )->float:

        maxDelta = np.inf
        time  = 0

        old_maxDelta = 1

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
                              conc[snidx],
                              cd.h[cd.saturated])

            dcdt = np.zeros(conc.shape[0])
            
            dcdt[cd.saturated] = D*d2
            dcdt[cd.unsaturated] = dcdt[cd.unsaturated]

            conc[cd.saturated] += dcdt[cd.saturated]*dt
            # conc[cd.unsaturated] += dcdt[cd.unsaturated]
            conc[cd.unsaturated] += dcdt[unidx]*dt 

            conc[conc < 0] = 0

            maxDelta = entropy(conc[cd.saturated]) / cd.saturated.shape[0]

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
                    ncols : int = 5,
                    side_size : float = 3,
                    qscale : float = None ,
                    log : bool = True,
                    pltargs : dict = None,
                    ) -> Tuple:

    ncnt = cnt.values

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
        if log:
            vals = np.log2(vals + 2)
        if qscale is not None:
            if qscale > 0 and qscale < 1:
                vals_q = np.quantile(vals,qscale,interpolation = 'nearest')
                vals[vals > vals_q] = vals_q
            else:
                print('WARNING : {} is not a proper quantile value'.format(qscale),
                      'within range (0,1)')

        ax[ii].set_title('Gene : {} \nDiffusion Time : {:0.3f}'.format(cnt.columns[ii],
                                                                  times[ii]))
        high_ordr = np.argsort(vals)
        ax[ii].scatter(crd[:,0][high_ordr],
                       crd[:,1][high_ordr],
                       c = vals[high_ordr],
                       **_pltargs,
                      )
        clean_axes(ax[ii])

    return (fig,ax)

def get_eigen_dmat(vals : np.ndarray,
                   normalized : bool = True,
                   ) -> np.ndarray :

    n_samples = vals.shape[0]
    dmat = np.zeros((n_samples,n_samples))

    if not normalized:
        nrm = np.linalg.norm(vals,axis = 1)
        vals = eigenscores / nrm.reshape(-1,1)

    for ii in range(n_samples-1):
        u = vals[ii,:]
        for jj in range(ii+1,n_samples):
            v = vals[jj,:]
            dmat[ii,jj] = np.arccos(np.dot(u,v))
            dmat[jj,ii] = dmat[ii,jj]

    return dmat



def get_eigen( mat : np.ndarray,
               thrs : float = 0.99,
               )-> np.ndarray :

    # matrix is n_spots x n_genes
    x_hat = mat.T

    # x_hat is n_genes x n_spots
    # rows are variables
    # columns are samples

    # center the variables
    mu = np.mean(x_hat,axis=0, keepdims = True)
    x_hat = (x_hat - mu)
    x_hat[np.isnan(x_hat)] = 0

    # get covariance matrix
    cov = np.cov(x_hat, rowvar = False)

    # compute eigencomps
    evals,evecs = np.linalg.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    evecs *= -1

    cs = np.cumsum(evals)

    n_comps = np.argmax( cs / cs[-1] > thrs)

    evecs = evecs[:,0:n_comps+1]
    evecs /= np.linalg.norm(evecs,axis = 0)
    proj = np.dot(mat.T,evecs)

    return (evecs,proj)

def cluster_data(counts : np.ndarray,
                 n_base = 500,
                 n_projs = 100,
                 threshold : float = 0.9,
                 ):

    epats,projs = get_eigen(counts[:,0:n_base],
                            thrs = threshold)

    projs = projs[0:n_projs,:]
    projs /=  np.linalg.norm(projs,axis = 1).reshape(-1,1)

    n_patterns = epats.shape[1]

    print("[INFO] : Using {} eigenpatterns".format(n_patterns))


    # cidx = Cl(metric = 'precomputed',
    #           max_eps = np.inf,
    #           xi = 0.01,
    #           min_samples = 2).fit_predict(get_eigen_dmat(projs))

    cidx = ACl(n_clusters = n_patterns,
            affinity = 'precomputed',
            linkage = 'complete',
            ).fit_predict(get_eigen_dmat(projs))

    n_clusters = np.unique(cidx)
    n_clusters = n_clusters[n_clusters >= 0]
    n_clusters = n_clusters.shape[0]

    repr_patterns = {} 
    for cl in np.unique(cidx):
        av_loads = np.mean(projs[cidx == cl,:],axis = 0)
        rpat = np.dot(epats,av_loads)
        repr_patterns.update({cl:rpat})


    print("[INFO] : Identified {} clusters".format(n_clusters))

    return (cidx,repr_patterns)

def visualize_representative(patterns : Dict[int,np.ndarray],
                             crd : np.ndarray,
                             ncols : int,
                             log : bool = True,
                             side_size : float = 3,
                             pltargs : dict = None,
                             normalize : bool = True,
                             ):

    nrows = np.ceil(len(patterns) / ncols).astype(int)

    _pltargs = {'s':40,
                'edgecolor':'black',
                'cmap':plt.cm.PuRd,
                }

    if pltargs is not None:
        for k,v in pltargs.items():
            _pltargs[k] = v
            if k == 'cmap' and isinstance(k,str):
                _pltargs[k] = eval(v)

    figsize = (1.2 * ncols * side_size,
               1.2 * nrows * side_size)

    fig,ax = plt.subplots(nrows,
                          ncols,
                          figsize = figsize)
    ax = ax.flatten()

    for cl,vals in patterns.items():
        if log:
            vals = np.log2(vals + 2)

        ax[cl].scatter(crd[:,0],
                       crd[:,1],
                       c = vals,
                       **_pltargs,
                       )
        ax[cl].set_title("Repr. Pattern {}".format(cl))

    for ii in range(ax.shape[0]):
        ax[ii] = clean_axes(ax[ii])

    return fig,ax


def visualize_clusters(counts : np.ndarray,
                       genes : pd.Index,
                       crd : np.ndarray,
                       labels : np.ndarray,
                       ncols : int,
                       log : bool = True,
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
    uni_labels = np.sort(uni_labels[uni_labels >= 0])

    for k,lab in enumerate(uni_labels):

        pos = np.where(labels == lab)[0]

        nrows = np.ceil(pos.shape[0] / ncols).astype(int)

        figsize = (1.2 * ncols * side_size,
                   1.2 * nrows * side_size)

        vizlist.append(list(plt.subplots(nrows,
                                         ncols,
                                         figsize = figsize)))

        vizlist[-1][0].suptitle("Family {}".format(lab))

        vizlist[-1][1] = vizlist[-1][1].flatten()

        for ii in range(pos.shape[0]):
            vals = counts[:,pos[ii]]
            if log:
                vals = np.log2(vals + 2)

            vizlist[-1][1][ii].set_title("Gene : {}".format(genes[pos[ii]]))
            vizlist[-1][1][ii].scatter(crd[:,0],
                                       crd[:,1],
                                       c = vals,
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
    return re.sub(':|-|\.| |','',
                  str(datetime.datetime.now()))

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
                        cnt.shape[1]-n_examine-n_topexpr+ninter]])

        _,pval = fe(cmat)

        print()
        print("[TOP SPATIAL PATTERNS]: {} genes".format(n_examine))
        print("[TOP EXPRESSED] : {} genes".format(n_topexpr))
        print("[INTERSECTION] : {} genes ".format(ninter))
        print("[P-VALUE] : {} ".format(pval))

def visualize_cdf(diff : np.ndarray,
                  )-> Tuple[plt.Figure,plt.Axes]:

    cs = np.cumsum(np.sort(diff)[::-1])
    fig, ax = plt.subplots(1,1,figsize = (10,10))

    ax.plot(cs, '-',
            color = 'black',
            )

    # ax.set_aspect("equal")

    ax.set_title("ECDF of Diffusion values")
    ax.set_xlabel("Ranked Sample")
    ax.set_ylabel("Cummulative Sum")

    for sp in ax.spines.values():
        sp.set_visible(False)

    return (fig,ax)


def entropy(xx):
    xnz = xx[xx>0]
    xs = np.sum(xnz)
    xn = xnz / xs
    xl = np.log(xn) 
    return (-xl*xn).sum()


def morans(xx,ws):
    N = xx.shape[0]
    W = ws.sum()
    mu = xx.mean()

    den = np.sum((xx - mu)**2)
    nom = (xx - mu).reshape(-1,1) * (xx -mu ).reshape(1,-1) * ws
    nom = nom.sum()

    return N / W * nom / den
