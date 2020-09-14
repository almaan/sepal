#!/usr/bin/env python3

"""Models for diffusion simulation

Methods related to the simulation
and classes compatible with these


"""

import numpy as np
import pandas as pd
import os.path as osp
import sys

from abc import ABC,abstractmethod, abstractproperty,ABCMeta
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import lap


from joblib import Parallel, delayed
from multiprocessing import cpu_count


from typing import Tuple,Union,List, Optional

from sepal.utils import eprint,wprint,iprint
import sepal.utils as ut
from sepal.datasets import RawData

class CountData(ABC):
    """Count Data Abstract class

    Represents the count data
    holding the transcription
    profiles. Contains the methods
    necessary to simulate diffusion.

    Arguments:
    ---------
    data : RawData
        data to be used stored in a RawData
        object.
    nn : int
        number of neighbours
    normalize: bool
        if library-size normalization should
        be applied to respective spot
    eps : float
        allowed difference in distance from
        specified radius. Returns approximate
        neighbors.


    """
    def __init__(self,
                 data : RawData,
                 nn : int = 4,
                 normalize : bool = False, 
                 eps : float = 0.1,
                 )-> None:

        
        # set count data and coordinates
        self.cnt = data.cnt
        self.crd = data.crd
        self.nn = nn
        self.edges = np.array([])
        # set real observed coordinates
        # for visualization
        self.real_crd = data.crd

        # format coordinates
        # normlizes distances
        self._format_crd()
        # update object specifications
        self._update_specs()

        # set allowed radius
        self.r = 1
        # set tolerance
        self.eps = eps
        # set grid size
        self.h = self.r * np.ones(self.S,) 
        # set the order of the edges 
        self._set_edges()

        # create a KDTree
        self.kdt = KDTree(self.crd)
        # update which locations that
        # have full neighborhood
        # and which have not
        self._remove_unsaturated()

    
    def _update_specs(self,
                      )->None:
        """Update specifications"""

        # set number of locations
        self.S = self.cnt.shape[0]
        # set number of profiles
        self.G = self.cnt.shape[1]


    def _scale_crd(self,
                   )->None:

        """Internally scale distances
        to allow for different
        array specifications

        """
        dists = cdist(self.crd,self.crd)
        dists[dists == 0] = np.inf 
        mn =  dists.min() 
        self.crd = self.crd / mn

    def normalize(self,
                  )-> None:

        """Library size normalization

        """

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
        """Update location saturation

        Identifies which locations that
        have a full neighborhood (saturated)
        or incomplete neighborhood (unsaturated)
        respectively. Unsaturated locations
        are considered as boundary points.


        """

        # lists to be filled in
        self.saturated  = np.array([],
                                   dtype=int)
        self.unsaturated = np.array([],
                                    dtype=int)

        # find neighbors for every spot
        for spot in range(self.S):
            nbr = self.kdt.query_ball_point(self.crd[spot,:],
                                            self.r + self.eps,
                                            )
            # if full neighborhood consuder saturated
            if len(nbr) >= self.nn + 1:
                self.saturated = np.append(self.saturated,int( spot ))
            # else add to unsaturated
            else:
                self.unsaturated = np.append(self.unsaturated,int( spot ))

        # set data frame indices of saturated and unsaturated
        # spots
        self.saturated_idx = self.cnt.index[self.saturated]
        self.unsaturated_idx = self.cnt.index[self.unsaturated]

    def get_satnbr_idx(self,
                       sel : np.ndarray,
                       ) -> np.ndarray:
        """get neighbors of saturated
        locations

        Parameters:
        ----------

        sel : np.ndarray
            spot indices to get neighbors for

        Returns:
        -------
        Neighbors of selected locations
        ordered by their relative position
        to the locations

        """

        # check if specified locations
        # are saturated
        _sat_set = set(self.saturated)
        _sel_set = set(sel)
        size_inter = len(_sat_set.intersection(_sel_set))
        if len(_sel_set) != size_inter:
            wprint("selected locations are not all saturated")

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

        """get neighbors of unsaturated
        locations.

        For each unsaturated location
        it will return the index
        of the nearest saturated
        point

        Parameters:
        -----------
        sel : np.ndarray
            array with indices of unsaturated
            locations to get neighbors for

        Returns:
        -------
        Indices of nearest saturated
        locations for the queried unsaturated
        locations.

        """

        # check if specified locations
        # are saturated
        _uns_set = set(self.unsaturated)
        _sel_set = set(sel)
        size_inter = len(_uns_set.intersection(_sel_set))
        if len(_sel_set) != size_inter:
            wprint("selected locations are not all unsaturated")

        dmat = cdist(self.crd,self.crd)
        dmat = dmat[sel,:]
        dmat[:,self.unsaturated] = np.inf
        dmat[:,sel] = np.inf

        
        nbrs = np.argsort(dmat,axis = 1)
        nbrs = nbrs[:,0]
        
        return nbrs.astype(int)


    def _getpos(self,
                origin_idx : int,
                nbr_idx : int,
                )-> int:

        """get edge position
        for a neighbor to a specified
        location.

        Parameters:
        ----------
        origin_idx : int
            index of location used as reference
        nbr_idx : int
           index of neighbor to reference 

        Returns:
        -------
        Indicator of which edge
        the neighbor represents

        """

        # compute difference between
        # reference and neighbor coordinate
        # vectors
        vec =  self.crd[nbr_idx,:] - self.crd[origin_idx,:] 
        # normalize to unit norm
        vec = vec / np.linalg.norm(vec)
        # compute angle between
        # differnece vector and
        # x-axis : span([1,0]^T)
        theta = np.arccos(vec[0])
        # map theta in [-pi,pi]
        # to theta in [0,2pi]
        if vec[1] < 0:
            theta = 2*np.pi - theta

        # find edge which neighbor
        # most likely represent
        pos = np.argmin(np.abs(self.edges - theta))
        # make sure integer
        pos = pos.astype(int)
        return pos

    @abstractmethod
    def _set_edges(self,
                   ) -> None:
        """determines the angle between
        a point and its respective neighbors
        """
        pass
    
    @abstractmethod
    def _format_crd(self,
                   ) -> None:
        """format coordinates
        usually internal scaling
        """
        pass

    @abstractmethod
    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:
        """definition of laplacian

        """
        pass

class ST1K(CountData):
    def __init__(self,
                 cnt : RawData,
                 normalize : bool = False, 
                 eps : float = 0.1,
                 )-> None:
        """ST1k count data class

        Derivative of CountData used to hold
        ST1k array based data


        data : RawData
            data to be used stored in a RawData
            object.
        normalize: bool
            if library-size normalization should
            be applied to respective spot
        eps : float
            allowed difference in distance from
            specified radius. Returns approximate
            neighbors.

        """

        self.nn = 4
        super().__init__(cnt,
                         self.nn,
                         normalize,
                         eps,)

    def _format_crd(self,
                )->None:
        """format coordinates"""

        self.crd = self.crd.round(0)
        self._scale_crd()

    def _set_edges(self,
                   )->None:
        """determine edges

        The edges are set to be arranged
        with pi/2 radians apart along,
        parallel to the coordinate
        axis.

        """
        self.edges = np.array([np.pi / 2 * n for \
                               n in range(4)])

    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:

        """second order approxiation
        of the laplacian. Uses
        five point stencil.

        Parameters:

        centers : np.ndarray
            values at center (reference)
        nbrs : np,ndarray
            values at neighboring spots
        h : np.ndarray
            step size

        Returns:
        -------
        Numerical approximation of laplacian
        at each location

        """
        
        return laplacian_rect(centers,nbrs,h)


class VisiumData(CountData):
    """Visium count data class

    Derivative of CountData used to hold
    Visium array based data


    data : RawData
        data to be used stored in a RawData
        object.
    normalize: bool
        if library-size normalization should
        be applied to respective spot
    eps : float
        allowed difference in distance from
        specified radius. Returns approximate
        neighbors.

    """


    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = False, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 6
        super().__init__(cnt,
                         self.nn,
                         normalize,
                         eps)

    def _format_crd(self,
                    )->None:
        """format coordinates"""
        self._scale_crd()

    def _set_edges(self,
                   )->None:
        """determine edges

        The edges are set to be arranged
        with pi/3 radians apart. See
        supplementary figure 1
        for description of arrangement

        """


        self.edges = np.pi / 6 + \
            np.array([n * np.pi / 3 for n in range(6)])
        ordering = np.array([5, 2, 1, 4, 3, 0])
        self.edges = self.edges[ordering]

    def laplacian(self,
                    centers : np.ndarray,
                    nbrs : np.ndarray,
                    h : np.ndarray,
                    )-> np.ndarray:

        """second order approximation
        of the laplacian. Uses
        seven point stencil.

        Parameters:

        centers : np.ndarray
            values at center (reference)
        nbrs : np,ndarray
            values at neighboring spots
        h : np.ndarray
            step size

        Returns:
        -------
        Numerical approximation of laplacian
        at each location

        """

        return laplacian_hex(centers,nbrs,h)

class ST2K(CountData):
    """ST2k count data class

    Derivative of CountData used to hold
    ST2k array based data


    data : RawData
        data to be used stored in a RawData
        object.
    normalize: bool
        if library-size normalization should
        be applied to respective spot
    eps : float
        allowed difference in distance from
        specified radius. Returns approximate
        neighbors.

    """
    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 4
        super().__init__(cnt,
                         self.nn,
                         normalize,
                         eps)

    def _format_crd(self,
                )->None:
        """format coordinates"""

        self._scale_crd()

    def _set_edges(self,
                   )->None:

        """determine edges

        The edges are set to be arranged
        with pi/2 radians apart along, with
        pi/4 degrees from respective
        coordinate axis

        """

        self.edges = np.pi / 4 + \
            np.array([np.pi / 2 * n for n in range(4)])
        
        order = np.array([0,2,1,3])
        self.edges = self.edges[order]


    def laplacian(self,
                  centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : np.ndarray,
                  )-> np.ndarray:

        """second order approxiation
        of the laplacian. Uses
        five point stencil.

        Parameters:

        centers : np.ndarray
            values at center (reference)
        nbrs : np,ndarray
            values at neighboring spots
        h : np.ndarray
            step size

        Returns:
        -------
        Numerical approximation of laplacian
        at each location

        """

        return laplacian_rect(centers,nbrs,h)



class UnstructuredData(CountData):
    """Unstructured count data class

    Derivative of CountData used to hold
    Unstructured data. Will first
    injectively map unstructured locations
    to a strucutred grid and then treat
    as a ST1k array.


    data : RawData
        data to be used stored in a RawData
        object.
    normalize: bool
        if library-size normalization should
        be applied to respective spot
    eps : float
        allowed difference in distance from
        specified radius. Returns approximate
        neighbors.

    """


    def __init__(self,
                 cnt : pd.DataFrame,
                 normalize : bool = True, 
                 eps : float = 0.1,
                 )-> None:

        self.nn = 4
        super().__init__(cnt,
                         self.nn,
                         normalize,
                         eps)

    def _format_crd(self,
                )->None:
        """format coordinates"""
        self._to_structured()
        self._scale_crd()

    def _set_edges(self,
                   )->None:
        """determine edges

        The edges are set to be arranged
        with pi/2 radians apart along,
        parallel to the coordinate
        axis.

        """
        self.edges = np.array([np.pi / 2 * n for n in range(4)])

    def _to_structured(self,
                        )->np.ndarray:
        """Transform unstructured data to structured

        Defines a structured grid over the
        domain which the unstructured
        data is observered within. Then
        formulates a LAP (Linear Assignment Problem) P.

        P is equivalent to minimizing the
        cost (defined by the distance) of moving
        a location to a newly defined grid point.

        Returns:
        -------
        Array with new mapped coordinates

        """

        # get range of unstructured data
        xmin,ymin = np.min(self.crd,axis = 0) 
        xmax,ymax = np.max(self.crd,axis = 0) 

        # generate grid over domain
        npoints = np.ceil(np.sqrt(self.crd.shape[0]))
        npoints = npoints.astype(int) 
        
        xx = np.linspace(xmin,xmax,npoints)
        yy = np.linspace(ymin,ymax,npoints)

        XX,YY = np.meshgrid(xx,yy)
        gx = XX.reshape(-1,1)
        gy = YY.reshape(-1,1)
        gcrd = np.hstack((gx,gy))

        # compute cost matrix
        dmat = cdist(self.crd,
                     gcrd,
                     metric = 'euclidean')

        # solve LAP
        _,cidxs,ridxs = lap.lapjv(np.exp(dmat / dmat.max()),
                                  extend_cost = True)
        # get new coordinates
        ncrd = gcrd[cidxs,:]

        # scale coordinates
        delta_x = np.diff(xx)[0]
        delta_y = np.diff(yy)[0]


        ncrd[:,0] = (ncrd[:,0] - xmin ) / delta_x
        ncrd[:,1] = (ncrd[:,1] - ymin ) / delta_y

        # compute step sizes
        h = dmat[np.arange(dmat.shape[0]),cidxs]
        h = h / h.max()
        h = h.reshape(-1,)

        # update stepsize and
        # coordinates
        self.crd = ncrd
        self.h = h

    def laplacian(self,
                centers : np.ndarray,
                nbrs : np.ndarray,
                h : np.ndarray,
                )-> np.ndarray:


        """second order approxiation
        of the laplacian. Uses
        five point stencil.

        Parameters:

        centers : np.ndarray
            values at center (reference)
        nbrs : np,ndarray
            values at neighboring spots
        h : np.ndarray
            step size

        Returns:
        -------
        Numerical approximation of laplacian
        at each location

        """
        return laplacian_rect(centers,
                              nbrs,
                              h)

def laplacian_rect(centers : np.ndarray,
                   nbrs : np.ndarray,
                   h : float,
                   )-> np.ndarray:
    """Laplacian approx rectilinear grid"""

    d2f = nbrs.sum(axis = 1) - 4*centers
    d2f = d2f / h**2

    return d2f

def laplacian_hex(centers : np.ndarray,
                  nbrs : np.ndarray,
                  h : float,
                  )-> np.ndarray:
    """Laplacian approx hexagonal grid"""
    d2f = nbrs.sum(axis = 1) - 6*centers
    d2f = d2f / h**2 * 2 / 3

    return d2f


def propagate(cd : CountData,
              thrs : float = 1e-8,
              dt : float = 0.001,
              stopafter : int = int(1e10),
              normalize : bool = True,
              diffusion_rate : Union[float,np.ndarray] = 1.0,
              num_workers : Optional[int] = None,
              scale : bool = False,
              pseudocount : float = 2,
              )-> pd.DataFrame:

    """Simulate Diffusion

    Simulates diffusion by propagating
    the system in time using Fick's
    second law of diffusion

    cd : CountData
       count data object 
    thrs : float
        threshold for convergence
    dt : float
        timestep
    stopafter : int 
        stop after given number of iterations
        if covergence is not reached
    normalize : bool
       normalize system, highly recommended
    diffusion_rate : Union[float,np.ndarray]
        diffusion rate (D). If float
        then homogenous over whole are,
        if array then each entry should
        specify the diffusion rate
        at a given location.
    num_workers : int
        number of workers to use
        if none is given maximal
        number is used
    scale : bool
        do minmax scaling
    pseudocount : float
        pseudocount to use in normalization

    Returns:
    -------
    The diffusion time for
    each profile

    """


    if num_workers is None:
        num_workers = int(cpu_count())
    else:
        num_workers = min(num_workers,
                          cpu_count())

    iprint("Using {} workers".format(num_workers))

    diff_prop = {'D':diffusion_rate,
                 'thrs':thrs,
                 'dt':dt,
                 'stopafter':stopafter}

    n_saturated = cd.saturated.shape[0]
    if n_saturated < 1:
        eprint("No Saturated spots")
        sys.exit(-1)
    else:
        iprint("Saturated Spots : {}".format(n_saturated))


    # stabilizing normalization
    if normalize:
        ncnt = cd.cnt.values
        ncnt = ut.normalize_expression(ncnt,c = pseudocount)
        colMax = np.max(np.abs(ncnt),axis = 0).reshape(1,-1)
        ncnt = np.divide(ncnt,
                         colMax,
                         where = colMax > 0)
        ncnt = ncnt.astype(float)
    else:
        ncnt = cd.cnt.values.astype(float)


    # get neighbor indices
    snidx = cd.get_satnbr_idx(cd.saturated)
    unidx = cd.get_unsatnbr_idx(cd.unsaturated)

    # Propagate in time
    try:
        # will use tqdm progress bar
        # if package installed
        from tqdm import tqdm
        iterable = tqdm(range(cd.G))
    except ImportError:
        iterable = range(cd.G)
    # spread on multiple workers
    times = Parallel(n_jobs=num_workers)(delayed(stepping)(idx,
                                                 ncnt[:,idx],
                                                 cd,
                                                 snidx,
                                                 unidx,
                                                 **diff_prop) for \
                                         idx in iterable)
    times = np.array(times)
    if scale:
        mn = times.min()
        mx = times.max()
        scaled = (times - mn) / (mx - mn)
        times = pd.DataFrame(scaled.reshape(-1,1),
                             columns = ['average'],
                             index = cd.cnt.columns,
                             )
    else:
        times = pd.DataFrame(times.reshape(-1,1),
                             columns = ['average'],
                             index = cd.cnt.columns,
                             )
    return times


def stepping(idx : int,
             conc : np.ndarray,
             cd : CountData,
             snidx : np.ndarray,
             unidx : np.ndarray,
             thrs : float,
             D : Union[float,np.ndarray],
             dt : float,
             stopafter : int,
             )->float:

        """Time stepping
        
        idx : int
            index of profile
        conc : np.ndarray
            concentration values
        cd : CountData
            count data object
        snidx : np.ndarray
            index of neighbors to saturated spots
        unidx : np.ndarray
            index of neighbors to unsaturated spots
        thrs : float
            threshold for convergence
        D : Union[float,np.ndarray]
            
        dt : float,
            timestep
        stopafter : int 
            stop after given number of iterations
            if covergence is not reached
        Returns:
        -------
        time for profile
        to reach convergence
            
        """

        time  = 0.0
        new_H = 1
        old_H = 0

        while np.abs(new_H - old_H) >\
              thrs and conc[cd.saturated].sum() > 0:
            # stop if convergence is not reached
            if time / dt > stopafter:
                genename = cd.cnt.columns[idx]
                wprint("Gene :"
                      "{} did not converge"
                      "".format(genename))
                break

            # update old entropy value
            old_H = new_H
            # update time
            time +=dt

            # get laplacian
            d2 = cd.laplacian(conc[cd.saturated],
                              conc[snidx],
                              cd.h[cd.saturated])

            # update concentration values
            dcdt = np.zeros(conc.shape[0])

            dcdt[cd.saturated] = D*d2
            dcdt[cd.unsaturated] = dcdt[cd.unsaturated]

            conc[cd.saturated] += dcdt[cd.saturated]*dt
            conc[cd.unsaturated] += dcdt[unidx]*dt
            # set values below zero to 0
            conc[conc < 0] = 0
            # compute entropy
            new_H = entropy(conc[cd.saturated]) / cd.saturated.shape[0]

        return time


def entropy(xx : np.ndarray,
            )->float:
    """Entropy of array 

    Elements in the array are
    normalized to represent
    probability values. Then
    Entropy are computed according
    to the definitionby Shannon

    S = -sum_i p_*log(p_i)

    Parameters:
    ----------
    xx : np.ndarray
        array for which entropy should
        be calculated

    Returns:
    -------
    Entropy of array

    """
    xnz = xx[xx>0]
    xs = np.sum(xnz)
    xn = xnz / xs
    xl = np.log(xn)
    return (-xl*xn).sum()
