#!/usr/bin/env python3

from enum import Enum
from copy import deepcopy
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

from typing import Tuple

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

def laplacian_rectilinear(centers : np.ndarray,
              nbrs : np.ndarray,
              h : float,
              )-> np.ndarray:

    d2f = nbrs.sum(axis = 1) - 4*centers
    d2f = d2f / h**2

    return d2f

def laplacian_hexagonal(centers : np.ndarray,
                        nbrs : np.ndarray,
                        h : float,
                        )-> np.ndarray:

    d2f = nbrs.sum(axis = 1) - 6*centers
    d2f = d2f / h**2 * 2 / 3

    return d2f

def filter_genes(mat,
                 min_occur : int = 5,
                 min_expr : int = 0,
                 )->None:

    keep_genes = (np.sum(mat.values > 0, axis = 0) > min_occur).astype(int)
    keep_genes *= (np.sum(mat.values, axis = 0) > min_expr).astype(int)
    keep_genes *= np.array([not bool(re.match('^RP|^MT',x.upper())) for x in mat.columns]).astype(int)
    mat = mat.iloc[:,keep_genes.astype(bool)]
    return mat

class CountData:
    def __init__(self,
                 cnt : pd.DataFrame,
                 radius : float,
                 n_neighbours : int,
                 eps : float = 10e-4,
                 rotate : int = None,
                 )-> None:

        self.cnt = cnt
        self.crd = self.get_crd(cnt.index)

        self._update_specs()


        if rotate is not None:
            self.theta = np.deg2rad(rotate)
            self.rmat = np.array([[np.cos(self.theta),np.sin(self.theta)],
                                  [-np.sin(self.theta),np.cos(self.theta)]]).T
            self._rotate(self.rmat)

        else:
            self.rmat = np.eye(2)
            self.theta = 0

    
        self.r = radius
        self.eps = eps
        self.nn = n_neighbours


        self.kdt = KDTree(self.crd)

        self._remove_unsaturated()

    def get_crd(self,
                idx : pd.Index,
                )->np.ndarray:

        crd = np.array([[float(x) for \
                         x in y.split('x') ] for\
                        y in idx])

        crd = crd
        
        return crd

    def _rotate(self,
                rmat : np.ndarray,
                )->None:

        self.crd = np.dot(rmat,self.crd.T).T
        self.crd = self.crd
        newidx = pd.Index(['x'.join([str(self.crd[s,0]),
                                     str(self.crd[s,1])]) for \
                           s in range(self.S)])

        self.cnt.index = newidx

    def _derotate(self,):
        self._rotate(np.linalg.inv(rmat))
 
    def _update_specs(self,):
 
         self.S = self.cnt.shape[0]
         self.G = self.cnt.shape[1]

    def _remove_unsaturated(self,
                            )-> None:

        self.saturated = []
        self.unsaturated = []
        for spot in range(self.S):
            nbr = self.kdt.query_ball_point(self.crd[spot,:],
                                            self.r + self.eps)

            if len(nbr) >= self.nn + 1:
                self.saturated.append(spot)
            else:
                self.unsaturated.append(spot)
                
        self.saturated = np.array(self.saturated).astype(int)
        self.saturated_idx = self.cnt.index[self.saturated]
        self._update_specs()

    def normalize(self,)-> None:
        vs = self.cnt.values.astype(float)
        sm = vs.sum(axis = 1)
        iszero = (sm == 0)
        if iszero.sum() > 0:
            vs[sm == 0,:] = np.nan
            vs = vs / sm.reshape(-1,1)
            vs[np.isnan(vs)] = 0
            self.cnt.iloc[:,:] = vs
        else:
            self.cnt.iloc[:,:] = vs/sm.reshape(-1,1)

    def get_allnbr_idx(self,
                       sel : np.ndarray,
                        ) -> np.ndarray:

        if self.nn == 4:
            nbrs = self.kdt.query_ball_point(self.crd[sel,:],
                                             self.r + self.eps)

            narr = np.nan * np.ones((sel.shape[0],self.nn))
            
            for k,spot in enumerate(sel):
                for n in range(0,self.nn+1):
                    if nbrs[k][n] != spot:
                        xdiff = np.abs(self.crd[nbrs[k][n],0] - self.crd[spot,0])
                        ydiff = np.abs(self.crd[nbrs[k][n],1] - self.crd[spot,1])
                        if xdiff > ydiff:
                            if self.crd[nbrs[k][n],0] < self.crd[spot,0]:
                                narr[k,0] = nbrs[k][n]
                            else:
                                narr[k,1] = nbrs[k][n]
                                
                        else:
                            if self.crd[nbrs[k][n],1] < self.crd[spot,1]:
                                narr[k,2] = nbrs[k][n]
                            else:
                                narr[k,3] = nbrs[k][n]

            return narr.astype(int)
        elif self.nn == 6:
            eps = 0.2 # 11degs
            for k,spot in enumerate(sel):
                for n in range(0,self.nn+1):
                    if nbrs[k][n] != spot:
                        nvec =  self.crd[spot,:] - self.crd[nbrs[k][n],:]
                        nvec = nvec / np.linalg.norm(nvec)
                        if nvec[0] < 0:
                            theta = np.arccos(-1*nvec[0])
                            if np.abs(theta) < eps:
                                narr[k,1] = nbrs[k][n]
                            elif theta < np.pi/2 - eps and theta > eps:
                                narr[k,3] = nbrs[k][n]
                            else:
                                narr[k,5] = nbrs[k][n]
                        else:
                            theta = np.arccos(nvec[0])
                            if np.abs(theta) < eps:
                                narr[k,0] = nbrs[k][n]
                            elif theta < np.pi/2 - eps and theta > eps:
                                narr[k,2] = nbrs[k][n]
                            else:
                                narr[k,4] = nbrs[k][n]
            return narr.astype(int)
                        
        else:
            print(f"Not implemented for this type of array")
            return None

    def get_allnbr_cnt(self,
                      )-> Tuple[np.ndarray,...]:

        idxs = self.get_allnbr_idx(self.saturated)

        return (self.cnt.values[self.saturated,:],self.cnt.values[idxs,:])

@print_params
def propagate(cd : CountData,
              thrs : float = 10e-8,
              dt : float = 0.1,
              stopafter : int = 10e5,
              normalize : bool = True,
              diffusion_rate : int = 1,
              )-> np.ndarray:
     

    if normalize:
        rowsums = np.sum(cd.cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cd.cnt.values,rowsums,where = rowsums > 0)
    else:
        ncnt = cd.cnt.values
    
    # Get neighbour indices
    nidx = cd.get_allnbr_idx(cd.saturated)

    D = diffusion_rate
    n_genes =  cd.G 
    times = np.zeros(n_genes)

    if cd.nn == 4:
        laplacian = laplacian_rectilinear
    elif cd.nn == 6:
        laplacian = laplacian_hexagonal

    # Propagate in time
    for gene in range(n_genes):
        conc = ncnt[:,gene].astype(float)
        conc[cd.unsaturated] = 0
        maxDelta = np.inf
        time = 0
        while maxDelta > thrs :
            if time / dt > stopafter:
                genename = cd.cnt.columns[gene]
                print("WARNING : Gene : {} did not convege".format(genename))
                break
            time +=dt
            d2 = laplacian(conc[cd.saturated],conc[nidx],1)
            dcdt = D*d2
            conc[cd.saturated] = conc[cd.saturated] +  dcdt*dt
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

def visualize_genes(cd : CountData,
                    times : np.ndarray,
                    n_genes : int = 20,
                    ncols : int = 5,
                    side_size : float = 3,
                    qscale : float = None ,
                    normalize : bool = True,
                    pltargs : dict = None,
                    ) -> Tuple:


    if normalize:
        rowsums = np.sum(cd.cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cd.cnt.values,rowsums,where = rowsums > 0)
    else:
        ncnt = cd.cnt.values

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

    for ii in range(n_genes):
        vals = ncnt[:,topgenes[ii]].reshape(-1,)
        if qscale is not None:
            if qscale > 0 and qscale < 1:
                vals_q = np.quantile(vals,qscale)
                vals[vals > vals_q] = vals_q
            else:
                print('WARNING : {} is not a proper quantile value'.format(qscale),
                      'within range (0,1)')

        ax[ii].set_title('{}'.format(cd.cnt.columns[topgenes[ii]]))
        ax[ii].scatter(cd.crd[:,0],
                       cd.crd[:,1],
                       c = vals,
                       **_pltargs,
                      )
        clean_axes(ax[ii])

    return (fig,ax)

