#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from scipy.spatial import KDTree

from typing import Tuple

def read_file(pth : str,
              index_col = 0,
              )->pd.DataFrame:
    df = pd.read_csv(pth,
                     sep = '\t',
                     header = 0,
                     index_col = index_col,
                     )
    return df


class NeighbourSpecs(Enum):
    VISIUM = 4
    OLD = 4

class RotateSpecs(Enum):
    VISIUM = 45
    OLD = 0

class RadiusSpecs(Enum):
    VISIUM = np.sqrt(2)
    OLD = 1.0


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
        for spot in range(self.S):
            nbr = self.kdt.query_ball_point(self.crd[spot,:],
                                            self.r + self.eps)

            if len(nbr) >= self.nn + 1:
                self.saturated.append(spot)
                
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
        else:
            print(f"Not implemented for this type of array")
            return None

    def get_allnbr_cnt(self,
                      )-> Tuple[np.ndarray,...]:

        idxs = self.get_allnbr_idx(self.saturated)

        return (self.cnt.values[self.saturated,:],self.cnt.values[idxs,:])
