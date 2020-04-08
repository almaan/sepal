#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os.path as osp

from abc import ABC,abstractmethod, abstractproperty
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import lap

import matplotlib.pyplot as plt

from typing import Tuple

from datasets import RawData

class CountData(ABC):
    def __init__(self,
                 data : RawData,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:


        self.cnt = data.cnt
        self.crd = data.crd

        self.real_crd = data.crd

        self._format_crd()
        self._update_specs()

        self.r = 1
        self.eps = eps
        self.h = self.r * np.ones(self.S,) 
        
        self._set_edges()

        self.kdt = KDTree(self.crd)
        self._remove_unsaturated()

    def _update_specs(self,
                      )->None:

        self.S = self.cnt.shape[0]
        self.G = self.cnt.shape[1]

    def get_crd(self,
                idx : pd.Index,
                )->np.ndarray:

        crd = np.array([[float(x.replace('X','')) for \
                        x in y.split('x') ] for\
                        y in idx])
        return crd

    def _scale_crd(self,
                   )->None:

        dists = cdist(self.crd,self.crd)
        dists[dists == 0] = np.inf 
        mn =  dists.min() 
        self.crd = self.crd / mn

    def normalize(self,
                  )-> None:

        vs = self.cnt.values.astype(float)
        sm = vs.sum(axis=1)
        iszero = (sm == 0)
        if iszero.sum() > 0:
            vs[sm == 0,:] = np.nan
            vs = vs / sm.reshape(-1,1)
            vs[np.isnan(vs)] = 0
        else:
            vs = vs / sm.reshape(1,-1)

        self.cnt = pd.DataFrame(vs,
                                index =  self.cnt.index,
                                columns = self.cnt.columns)

    def _remove_unsaturated(self,
                            )-> None:

        self.saturated = []
        self.unsaturated = []

        for spot in range(self.S):
            nbr = self.kdt.query_ball_point(self.crd[spot,:],
                                            self.r + self.eps,
                                            )
            if len(nbr) >= self.nn + 1:
                self.saturated.append(spot)
            else:
                self.unsaturated.append(spot)
                
        self.saturated = np.array(self.saturated).astype(int)
        self.saturated_idx = self.cnt.index[self.saturated]

    def get_satnbr_idx(self,
                       sel : np.ndarray,
                       ) -> np.ndarray:

        nbrs = self.kdt.query_ball_point(self.crd[sel,:],
                                        self.r + self.eps)

        narr = np.nan * np.ones((sel.shape[0],self.nn))

        for k,spot in enumerate(sel):
            for n in range(0,self.nn+1):
                    if nbrs[k][n] != spot:
                        pos = self._getpos(spot,nbrs[k][n])
                        narr[k,pos] = nbrs[k][n]

        return narr.astype(int)

    def get_unsatnbr_idx(self,
                       sel : np.ndarray,
                       ) -> np.ndarray:

        dmat = cdist(self.crd,self.crd)
        dmat = dmat[sel,:]
        dmat[:,sel] = np.inf

        
        nbrs = np.argsort(dmat,axis = 1)
        nbrs = nbrs[:,0]
        
        return nbrs.astype(int)




    def get_allnbr_cnt(self,
                            )-> Tuple[np.ndarray,...]:

        idxs = self.get_allnbr_idx(self.saturated)

        return (self.cnt.values[self.saturated,:],
                self.cnt.values[idxs,:])

    def _getpos(self,
                origin_idx : int,
                nbr_idx : int,
                )-> int:

        vec =  self.crd[nbr_idx,:] - self.crd[origin_idx,:] 
        vec = vec / np.linalg.norm(vec)
        theta = np.arccos(vec[0])

        if vec[1] < 0:
            theta = 2*np.pi - theta

        pos = np.argmin(np.abs(self.edges - theta))
        pos = pos.astype(int)

        return pos

    @abstractmethod
    def _set_edges(self,
                   ) -> None:
        pass
    
    @abstractmethod
    def _format_crd(self,
                   ) -> None:
        pass

    @abstractmethod
    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:
        pass

    # @abstractmethod
    # def gradient(self,
    #              centers : np.ndarray,
    #              nbrs : np.ndarray,
    #              h: np.ndarray,
    #              ) -> np.ndarray:
    #     pass

class ST1K(CountData):
    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 4
        super().__init__(cnt,normalize,eps)

    def _format_crd(self,
                )->None:

        self.crd = self.crd.round(0)
        self._scale_crd()

    def _set_edges(self,
                   )->None:

        self.edges = np.array([np.pi / 2 * n for \
                               n in range(4)])

    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:

        return laplacian_rect(centers,nbrs,h)


class VisiumData(CountData):
    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 6
        super().__init__(cnt,normalize,eps)

    def _format_crd(self,
                    )->None:
        self._scale_crd()

    def _set_edges(self,
                   )->None:

        self.edges = np.pi / 6 + \
            np.array([n * np.pi / 3 for n in range(6)])
        ordering = np.array([5, 2, 1, 4, 3, 0])
        self.edges = self.edges[ordering]

    def laplacian(self,
                    centers : np.ndarray,
                    nbrs : np.ndarray,
                    h : np.ndarray,
                    )-> np.ndarray:

        return laplacian_hex(centers,nbrs,h)

class ST2K(CountData):

    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 4
        super().__init__(cnt,normalize,eps)

    def _format_crd(self,
                )->None:

        self._scale_crd()

    def _set_edges(self,
                   )->None:

        self.edges = np.pi / 4 + \
            np.array([np.pi / 2 * n for n in range(4)])
        
        order = np.array([0,2,1,3])
        self.edges = self.edges[order]


    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:

        return laplacian_rect(centers,nbrs,h)



class UnstructuredData(CountData):
    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 4
        super().__init__(cnt,
                         normalize,
                         eps)

    def _format_crd(self,
                )->None:

        self._to_structured()
        self._scale_crd()

    def _set_edges(self,
                   )->None:

        self.edges = np.array([np.pi / 2 * n for n in range(4)])

    def _to_structured(self,
                        )->np.ndarray:

            xmin,ymin = np.min(self.crd,axis = 0) 
            xmax,ymax = np.max(self.crd,axis = 0) 

            npoints = np.ceil(np.sqrt(self.crd.shape[0]))
            npoints = npoints.astype(int) 
            
            xx = np.linspace(xmin,xmax,npoints)
            yy = np.linspace(ymin,ymax,npoints)

            XX,YY = np.meshgrid(xx,yy)
            gx = XX.reshape(-1,1)
            gy = YY.reshape(-1,1)
            gcrd = np.hstack((gx,gy))

            dmat = cdist(self.crd,
                         gcrd,
                         metric = 'euclidean')

            _,cidxs,ridxs = lap.lapjv(np.exp(dmat / dmat.max()),
                                      extend_cost = True)
            ncrd = gcrd[cidxs,:]

            delta_x = np.diff(xx)[0]
            delta_y = np.diff(yy)[0]


            ncrd[:,0] = (ncrd[:,0] - xmin ) / delta_x
            ncrd[:,1] = (ncrd[:,1] - ymin ) / delta_y

            h = dmat[np.arange(dmat.shape[0]),cidxs]
            h = h / h.max()
            h = h.reshape(-1,)

            self.crd = ncrd
            self.h = h

    def laplacian(self,
                centers : np.ndarray,
                nbrs : np.ndarray,
                h : np.ndarray,
                )-> np.ndarray:

        return laplacian_rect(centers,nbrs,h)

def laplacian_rect(centers : np.ndarray,
                   nbrs : np.ndarray,
                   h : float,
                   )-> np.ndarray:

    d2f = nbrs.sum(axis = 1) - 4*centers
    d2f = d2f / h**2

    return d2f

def laplacian_hex(centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : float,
                  )-> np.ndarray:

    d2f = nbrs.sum(axis = 1) - 6*centers
    d2f = d2f / h**2 * 2 / 3

    return d2f


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

        ncnt = ut.normalize_expression(ncnt)

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
             D : Union[float,np.ndarray],
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


def entropy(xx):
    xnz = xx[xx>0]
    xs = np.sum(xnz)
    xn = xnz / xs
    xl = np.log(xn)
    return (-xl*xn).sum()

