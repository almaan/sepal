#!/usr/bin/env python3

"""Pattern Family function

Functions to generate and visualize
pattern families


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os.path as osp
import sys

from sklearn.cluster import AgglomerativeClustering as ACl
from sklearn.decomposition import PCA

from argparse import Namespace as ARGS
from typing import Tuple,Dict,Union,List,Any
from sepal.utils import iprint,eprint,wprint,VARS
import sepal.utils as ut
import sepal.models as m

def plot_representative(motifs : Dict[int,np.ndarray],
                        crd : np.ndarray,
                        ncols : int,
                        side_size : float = 300,
                        pltargs : dict = None,
                        normalize : bool = False ,
                        pseudocount : float = 2,
                        )->Tuple[plt.Figure,plt.Axes]:

    """Plot representative motifs

    Parameters:
    ----------

    motifs : Dict[int,np.ndarray]
        motif index and profile for respective
        family
    crd : np.ndarray
        coordinates to use. [n_locations x 2]
    ncols : int
        number of columns to use
    side_size : float
        side size of each subplot.
        Given in pixels
    pltargs : dict
        style dict for plot
    normalize : bool
        normalize motif expression. Strongly
        avoided since negative values may
        be present.
    pseudocount : float
        pseudocount to use in normalization

    Returns
    -------
    Tuple with figure and
    axes objects

    
    """

    side_size *= 0.01

    # determine number of rows
    nrows = np.ceil(len(motifs) / ncols).astype(int)

    _pltargs = VARS.PLTARGS

    if pltargs is not None:
        for k,v in pltargs.items():
            _pltargs[k] = v
            if k == 'cmap' and isinstance(k,str):
                try:
                    _pltargs[k] = eval("plt.cm." + v)
                except:
                    _pltargs[k] = plt.cm.magma

    figsize = (1.2 * ncols * side_size,
               1.2 * nrows * side_size)

    fig,ax = plt.subplots(nrows,
                          ncols,
                          figsize = figsize)
    ax = ax.flatten()

    for fl,vals in motifs.items():
        if normalize:
            vals = ut.normalize_expression(vals,
                                           c = pseudocount)

        ax[fl].scatter(crd[:,0],
                       crd[:,1],
                       c = vals,
                       **_pltargs,
                       )
        ax[fl].set_title("Repr. Motif {}".format(fl))

    for ii in range(ax.shape[0]):
        ut.clean_axes(ax[ii])

    return fig,ax



def plot_families(counts : np.ndarray,
                  genes : pd.Index,
                  crd : np.ndarray,
                  labels : np.ndarray,
                  ncols : int,
                  normalize : bool = True,
                  side_size : float = 300,
                  pltargs : dict = None,
                  split_title : list = None,
                  pseudocount : float = 2,
                  )->List[Tuple[plt.Figure,np.ndarray]]:

    """ Plot pattern families
    
    Parameters:
    ----------
    counts : np.ndarray
        count matrix with transcription profiles
    genes : pd.Index
        name of gens to include
    crd : np.ndarray
        coordinates. [n_locations x 2]
    labels : np.ndarray
        labels indicating which family respective
        profile belongs to
    ncols : int
        number of columns
    normalize : bool
       set to true if log transform
       (normalization) of expression
       values should be performed.
    side_size : float
        size of the side of each
        plotted profile. Unit is
        pixels.
    pltargs : dict
        dictionary with style
        features for plotting
    split_title: Tuple[str,int]
        include if profile names should
        be splitted. First element is string
        to split by, second the element
        to keep.
    pseudocount : float
        pseudocount to use in normalization
    
    Returns
    -------

    Tuple of Figure and the corresponding
    axes objects containing visualization
    of the families

    """

    side_size *= 0.01

    # set aesthetics
    _pltargs = {'s':40,
                'edgecolor':'black',
                'cmap':plt.cm.PuRd,
                }

    if pltargs is not None:
        for k,v in pltargs.items():
            _pltargs[k] = v
            if k == 'cmap' and isinstance(k,str):
                _pltargs[k] = eval("plt.cm."+ v)


    # generate visualizations for
    # each family
    vizlist : List[Tuple[plt.Figure,np.ndarray]] = []
    uni_labels = np.unique(labels)
    uni_labels = np.sort(uni_labels[uni_labels >= 0])

    for k,lab in enumerate(uni_labels):

        pos = np.where(labels == lab)[0]

        nrows = np.ceil(pos.shape[0] / ncols).astype(int)

        figsize = (1.2 * ncols * side_size,
                   1.2 * nrows * side_size)


        _fig, _ax = plt.subplots(nrows,
                                  ncols,
                                  figsize = figsize)
        _ax = _ax.flatten()
        _fig.suptitle("Family {}".format(lab))

        vizlist.append((_fig,_ax))


        for ii in range(pos.shape[0]):
            vals = counts[:,pos[ii]]
            if normalize:
                vals = ut.normalize_expression(vals,
                                               c = pseudocount)

            title = genes[pos[ii]]
            if split_title is not None:
                title = title.split(split_title[0])[int(split_title[1])]

            vizlist[-1][1][ii].set_title("Gene : {}".format(title),
                                         fontsize = 15)
            vizlist[-1][1][ii].scatter(crd[:,0],
                                       crd[:,1],
                                       c = vals,
                                       **_pltargs,
                                       )

            vizlist[-1][1][ii].set_aspect('equal')
            ut.clean_axes(vizlist[-1][1][ii])

        for jj in range(ii+1,ncols*nrows):
            ut.clean_axes(vizlist[-1][1][jj])
            vizlist[-1][1][jj].set_visible(False)

    return vizlist


def get_eigen_dmat(vals : np.ndarray,
                   ) -> np.ndarray :
    """compute distance
    matrix in eigenpattern space.
    If the eigenpattern space has
    dimension |K| and |P| profiles
    are analyzed, vals should have
    the shape |P|x|K|

    Parameters:
    ----------

    vals : np.ndarray
       expression profiles projections
       onto eigenpattern space. Given
       w.r.t. to eigenpattern basis.

    Returns:
    -------
    Returns distance matrix
    for expression profiles
    projections. Shape is
    |P|x|P|


    """

    # unit normalize vectors 
    nrm = np.linalg.norm(vals,axis = 1,
                         keepdims = True)
    vals = ut.safe_div(vals,
                     nrm,
                    )

    # compute inner product
    dmat = np.dot(vals,vals.T)
    # adjust for numerical errors
    dmat[np.abs(dmat - 1) < 10e-6] = 1.0
    dmat[np.abs(dmat + 1) < 10e-6] = -1.0
    # compute angle between all vectors
    dmat = np.arccos(dmat)

    return dmat


def get_eigenpatterns( mat : np.ndarray,
                       thrs : float = 0.99,
                       )-> Tuple[np.ndarray,np.ndarray] :
    """Compute eigenpatterns 

    For a set of transcription profiles,
    compute the eigenpatterns that
    explain thrs amount of the
    variance in the data.

    The dimensions of mat is expected
    to be n_locations x n_profiles

    Parameters:
    ----------

    mat : np.ndarray
        matrix containing the expression profiles
    thrs : float
        percentage of the variance that should
        be explained by eigenpatterns

    Returns
    -------
    A tuple with first element being the
    eigenpatterns and the second being
    the expression profiles projected
    onto eigenpattern space, given
    in the basis constructed from the
    eigenpatterns


    """

    if thrs > 1:
        thrs = thrs * 0.01

    # treat genes as observations and
    # locations as features
    pca_fit = PCA().fit(mat.T)

    # compute number of components required
    expl_var = pca_fit.explained_variance_ratio_

    # get number of eigenpatterns rquired
    # to explain specified variance
    n_comps = np.argmax(np.cumsum(expl_var) > thrs)
    evecs = pca_fit.components_[0:n_comps+1,:]
    # make unit vectors
    norms = np.linalg.norm(evecs,
                            axis = 1,
                            keepdims=True)

    # normalize to unit length
    evecs = ut.safe_div(evecs,norms)

    # get projection w.r.t. eigenpattern
    # basis
    loads = np.dot(evecs,mat).T

    return (evecs,loads)


def get_families(raw_counts : np.ndarray,
                 n_base : int = 500,
                 n_sort : int = 100,
                 threshold : float = 0.9,
                 )->Tuple[np.ndarray,Dict[int,np.ndarray]]:

    """Get pattern families

    Sort transcription profiles into
    pattern families.


    Parameters:
    ----------

    counts : np.ndarray
        count matrix for the expression patterns.
        Dimensions ares (n_locations x n_profiles)
    n_base : int
        number of profiles to use upon extraction
        of eigenpatterns
    n_sort : int
        number of profiles to be assorted
        into pattern families
    threshold : float
        percentage of variance that should be
        explained by the eigenpatterns

    Returns:
    -------
    Tuple with first element being the
    family index of each profile and second
    element being a dictionary with the profile
    of each representative pattern.

    """

    counts = ut.safe_div(raw_counts,raw_counts.sum(axis=1,keepdims=True))

    # get eigenpatterns and projections
    epats,loads = get_eigenpatterns(counts[:,0:n_base],
                                    thrs = threshold)

    # only use n_sort profiles to
    # construct the families
    loads = loads[0:n_sort,:]

    # unit normalize
    norms = np.linalg.norm(loads,
                           axis = 1,
                           keepdims = True,
                           )
    nloads = ut.safe_div(loads,norms)


    n_patterns = epats.shape[0]
    iprint("Using {} eigenpatterns".format(n_patterns))


    fidx = ACl(n_clusters = n_patterns,
               affinity = 'precomputed',
               linkage = 'complete',
               ).fit_predict(get_eigen_dmat(nloads))

    n_families = np.unique(fidx).shape[0]

    # compute representative motifs
    motifs = {}
    for fl in np.unique(fidx):
        av_loads = np.mean(loads[fidx == fl,:],
                           axis = 0,
                           keepdims = True)

        rpat = np.dot(av_loads,epats).flatten()
        motifs.update({fl:rpat})


    iprint("Identified {} families".format(n_families))

    return (fidx,motifs)


def main(times_all :pd.DataFrame,
           cd : m.CountData,
           sampletag : str,
           args : ARGS,
           )->None:

    iprint("Assembling pattern families")

    # sort profiles by their rank metric
    sort_genes = np.argsort(times_all[VARS.SEL_COLUMN].values)[::-1]

    # select number of profiles
    # to construct eigenspace basis from
    if args.n_base_genes is None:
        args.n_base_genes = np.min((args.n_genes*2,
                                    times_all.shape[0]))
    else:
        args.n_base_genes = np.min((args.n_base_genes,
                                    times_all.shape[0]))

    # get sorted rank metric
    use_genes = times_all.index.values[sort_genes]
    # adjust threshold
    args.threshold = np.clip(args.threshold,0,100)
    # if percentage is given adjust
    # to fraction
    if args.threshold > 1:
        args.threshold *= 0.01

    # get families and representative motifs
    family_labels,repr_patterns = get_families(cd.cnt.loc[:,use_genes].values,
                                    n_base = args.n_base_genes,
                                    n_sort = args.n_genes,
                                    threshold = args.threshold,
                                    )
    # save family assortment
    families = pd.DataFrame(family_labels,
                            index = use_genes[0:args.n_genes],
                            columns = ['family'],
                            )
    
    out_fl_pth = osp.join(args.out_dir,'-'.join([sampletag,
                                                 'family',
                                                'index.tsv']
                                                 ))
    families.to_csv(out_fl_pth,
                    sep = '\t',
                    header = True,
                    index = True,
                    )
    # save representative patterns
    out_repr = pd.DataFrame(repr_patterns)
    out_repr.index = cd.cnt.index

    out_repr.to_csv(osp.join(args.out_dir,
                             sampletag + "-representative" +\
                             ".tsv"),
                             header = True,
                             index = True,
                             sep = '\t'
                             )

    # visualize results if specified
    if args.plot:

        reprviz,_ = plot_representative(repr_patterns,
                                           crd = cd.real_crd,
                                           ncols = args.n_cols,
                                           pltargs = args.style_dict,
                                           pseudocount = args.pseudocount,
                                           )

        reproname = osp.join(args.out_dir,''.join([sampletag,
                                                    '-representative.png',
                                                    ],
                                                    ))
        reprviz.savefig(reproname)

        family_plots = plot_families(cd.cnt.loc[:,use_genes[0:args.n_genes]].values,
                                     genes = use_genes[0:args.n_genes],
                                     crd = cd.real_crd,
                                     labels = family_labels,
                                     ncols = args.n_cols,
                                     pltargs = args.style_dict,
                                     split_title = args.split_title,
                                     side_size = args.side_size,
                                     pseudocount = args.pseudocount,
                                      )

        for fl in range(len(family_plots)):
            famoname = osp.join(args.out_dir,''.join([sampletag,
                                                      '-family-',
                                                       str(fl),
                                                       '.png'],
                                                     ))
            fig : plt.Figure = family_plots[fl][0]
            fig.savefig(famoname)

    return None


