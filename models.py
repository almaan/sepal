#!/usr/bin/env python3

from abc import ABC
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from typing import Tuple


class CountData(ABC):
    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:


        self._update_specs()
        self.cnt = cnt
        self.crd = self.get_crd(cnt.index)
        self.real_crd = self.get_crd(cnt.index)

        self.nn = None

        self.r = 1
        self.eps = eps
        self.h = self.r * np.ones(self.S,) 
        
        self._set_edges()

    def get_crd(self,
                idx : pd.Index,
                )->np.ndarray:

        crd = np.array([[float(x) for \
                        x in y.split('x') ] for\
                        y in idx])

        dists = cdist(self.crd,self.crd)
        dists[dists == 0] = dists.max()
        mn = np.min(dists)
        self.crd = self.crd / mn
        
        return crd


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

        self.cnt = pd.DataFrame(vs/sm.reshape(shape),
                                index =  self.cnt.index,
                                columns = self.cnt.columns)


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

    def get_allnbr_idx(self,
                       sel : np.ndarray,
                       ) -> np.ndarray:

        nbrs = self.kdt.query_ball_point(self.crd[sel,:],
                                        self.r + self.eps)

        narr = np.nan * np.ones((sel.shape[0],self.nn))

        for k,spot in enumerate(sel):
            for n in range(0,self.nn+1):
                pos = self._getpos(spot,nbrs[k][n])
                narr[k,pos] = nbrs[k][n]

        return narr.astype(int)



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
    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:


class ST1K(CountData):
    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        super().__init__(*args,**kwargs)
        self.nn = 4

    def _set_edges(self,
                   )->None:
        self.edges = np.array([np.pi / 2 * n for n in range(4)])

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

        super().__init__(*args,**kwargs)
        self.nn = 6

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
        super().__init__(*args,**kwargs)
        self.nn = 4


    def _set_edges(self,
                   )->None:

        self.edges = np.pi / 4 + \
            np.array([np.pi / 2 * n for n in range(4)])

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

        super().__init__(*args,**kwargs)
        self.nn = 4
        self.crd, self.h = self._to_structured

    def _set_edges(self,
                   )->None:

        self.edges = np.array([np.pi / 2 * n for n in range(4)])

    def _to_structured(self,
                        )->np.ndarray:

            xmin,ymin = np.min(self.crd,axis = 0) 
            xmax,ymax = np.max(self.crd,axis = 0) 

            npoints = np.ceil(np.sqrt(self.crd.shape[0])).astype(int)
            xx = np.linspace(xmin,xmax,npoints)
            yy = np.linspace(ymin,ymax,npoints)

            XX,YY = np.meshgrid(xx,yy)
            gx = XX.reshape(-1,1)
            gy = YY.reshape(-1,1)
            gcrd = np.hstack((gx,gy))

            dmat = cdist(self.crd,
                        gcrd,
                        metric = 'euclidean')

            _,cidxs,ridxs = lap.lapjv(np.exp(dmat),
                                      extend_cost = True)
            ncrd = gcrd[cidxs,:]

            delta_x = np.diff(xx)[0]
            delta_y = np.diff(yy)[0]


            ncrd[:,0] = (ncrd[:,0] - xmin ) / delta_x
            ncrd[:,1] = (ncrd[:,1] - ymin ) / delta_y

            h = dmat[np.arange(dmat.shape[0]),cidxs]
            h = h / h.max()
            h = h.reshape(-1,)

            return ncrd, h

        def gridify(self,)->None:
            self.oldcrd = self.crd[:,:]
            self.crd,self.h = self._to_structured()

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







