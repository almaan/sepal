#!/usr/bin/env python3

""" Utilities

Support functions and classes used
in sepal

"""

import re
import datetime

from sepal.__init__ import __version__

import time as Time
import re
import sys

import numpy as np
import pandas as pd

import kneed

import matplotlib.pyplot as plt
from matplotlib import rcParams

from scipy.ndimage import gaussian_filter
from typing import Tuple,Dict,Union,List

rcParams.update({'figure.max_open_warning': 0})

def custom_print(title : str,
                 s: str,
                 )->None:
    """custom print"""
    print("[{}] : {}".format(title,s))

def eprint(s : str,
           )->None:
    """error print"""
    custom_print("ERROR",s)

def wprint(s : str,
           )->None:
    """warning print"""
    custom_print("WARNING",s)

def iprint(s : str,
           )->None:
    """info print"""
    custom_print("INFO",s)

class VARS:
    """default variables"""
    SEL_COLUMN = 'average'
    PVAL_EPS = 1e-273
    PLTARGS = {'s':80,
               'edgecolor':'none',
               'cmap':plt.cm.magma,
               }




def normalize_expression(x : np.ndarray,
                         c : float = 2,
                         )->np.ndarray:
    """Normalization procedure

    Parameters:
    -----------
    x : np.ndarray
        data to be normalized

    Returns:
    -------
    vector with transformed data

    """
    return np.log2(x + c)


def safe_div(x : Union[np.ndarray,float],
             d : np.ndarray,
             ):
    """ safe division

    Divides x by d (denominator),
    at all instances except for where d = 0
    maintains the original dimensions of x

    Parameters
    ----------
    x : Union[np.ndarray,float]
       vector or number representing nominator
    d : np.ndarray
       vector to divide with, representing the
       denominator

    Returns:
    -------
    vector representing x/d excpet form
    instances where d = 0

    """
    return np.divide(x,
                     d,
                     where = (d.flatten() > 0).reshape(d.shape))

def read_file(pth : str,
              index_col : int = 0,
              )->pd.DataFrame:

    """standard function to read files

    Parameters
    ----------

    pth : str
        path to file
    index_col : int
        column to be used as index

    Returns:
    -------
    Pandas DataFrame with read file

    """

    df = pd.read_csv(pth,
                     sep = '\t',
                     header = 0,
                     engine = 'c',
                     index_col = index_col,
                     )
    return df

def filter_genes(mat : pd.DataFrame,
                 min_occur : int = 5,
                 min_expr : int = 0,
                 max_zero_percentage : float = 1.0,
                 filter_spurious : bool = True,
                 )->pd.DataFrame:

    """filter genes

    will filter a count matrix in place
    expects that the provided matrix has
    genes along columns and observations as
    rows

    mat : pd.DataFrame
        count matrix to be filtered
    min_occur : int
        minimal number of capture locations
        which a gene has to be observed at
    min_expr : minimal total number of
        observed features (taken over all)
        locations for a gene
    max_zero_fraction: float
        a value in [0,1] giving the
        upper bound for the fraction
        of zero observations for a gene
    filter_spurious: bool
        filter mitochondrial and
        ribosomal genes

    """

    keep_genes = (np.sum(mat.values > 0,
                         axis = 0) > min_occur).astype(int)

    keep_genes *= (np.sum(mat.values,
                        axis = 0) > min_expr).astype(int)

    max_zero_percentage = min(max_zero_percentage,1)

    n_tot = mat.shape[0]

    keep_genes *= (np.sum(mat.values == 0,
                          axis = 0,
                        ) / n_tot <= max_zero_percentage).astype(int)

    if filter_spurious:
        keep_genes *= np.array([not bool(re.match('^RP|^MT',x.upper())) \
                                for x in mat.columns]).astype(int)

    mat = mat.iloc[:,keep_genes.astype(bool)]

    return mat


def clean_axes(ax : plt.Axes,
                )->None:
    """cleans axes for aesthetic purposes"""

    ax.set_aspect('equal')
    ax.set_yticks([])
    ax.set_xticks([])
    for pos in ax.spines.keys():
        ax.spines[pos].set_visible(False)
    return None

def get_knee_point(y : np.ndarray,
                   x : np.ndarray = None,
                   sensitivity : float = 1.5,
                   )-> Union[int,float]:

    """Finds the inflection point

    Numerically finds the point of
    inflection for a set of observations
    y = f(x). If no x-values are provided
    these are taken as the index of
    the y-values. The function f
    is may be any arbitrary function.

    Parameters:
    ----------
    y : np.ndarray
       function values, y = f(x)
    x : np.ndarray
        independent variable from
        which y is generated by
    sigma : float
        bandwidth of kernel

    Returns:
    -------
    The value x_i s.t. f''(x_i) = 0

    """

    _kn = kneed.KneeLocator(x,
                            y,
                            S = sensitivity,
                            curve = "convex",
                            direction = "decreasing",
                            )

    ipoint = _kn.knee
    ipoint -= 1

    if x is not None:
        ipoint = x[ipoint]

    return ipoint

def plot_profiles(cnt : pd.DataFrame,
                    crd : np.ndarray,
                    rank_values : np.ndarray,
                    ncols : int = 5,
                    side_size : float = 350,
                    qscale : float = None ,
                    normalize : bool = True,
                    pltargs : dict = None,
                    split_title : Tuple[str,int] = None,
                    pval : bool = False,
                    pseudocount : float = 2,
                    ) -> Tuple[plt.Figure,plt.Axes]:

    """Visualize a set of transcription
    profiles.

    it is assumed that rank_values and cnt
    are matched. All values included in
    the expression data will be
    visualized

    Parameters:
    ----------
    cnt : pd.DataFrame
        expression data. Genes as columns
        capture locations as rows.
    crd : np.ndarray
        coordinates to use. [n_locations x 2]
    rank_values : np.ndarray
       values by which profiles are ranked. For
       example the diffusion times
    ncols : int
        number of columns to use
    side_size : float
        size of the side of each
        plotted profile. Unit is
        pixels.
    qscale : int
       cutoff for quantile scaling.
       If none provided, no quantile
       scaling is performed.
    normalize : bool
       set to true if log transform
       (normalization) of expression
       values should be performed.
    pltargs : dict
        dictionary with style
        features for plotting
    split_title: Tuple[str,int]
        include if profile names should
        be splitted. First element is string
        to split by, second the element
        to keep.
    pval : bool
        set to True if the rank_values
        correspond to p/q-values.

    Returns
    -------

    Tuple of Figure and the corresponding
    axes objects for the visualized profiles

    """

    # adjust side size to matplotlib untis
    side_size /= 100

    # get expression values
    ncnt = cnt.values

    # get number of genes
    n_genes = cnt.shape[1]

    # setup figure
    nrows = np.ceil(n_genes / ncols).astype(int)

    figsize = (1.2 * ncols * side_size,
               1.2 * nrows * side_size)

    fig,ax = plt.subplots(nrows,
                          ncols,
                          figsize=figsize)
    ax = ax.flatten()

    # define plot aesthetics
    _pltargs = {'s':80,
                'edgecolor':'none',
                'cmap':plt.cm.magma,
                }


    # set colormap
    use_rgba = False
    if pltargs is not None:
        for k,v in pltargs.items():
            _pltargs[k] = v
            if k == 'cmap':
                # if colormap is given in style dict
                if isinstance(v,str) and v != 'alpha':
                    # try to use specified colormap
                    try:
                        _pltargs[k] = eval("plt.cm." + v)
                    # default to magma if fail
                    except:
                        _pltargs[k] = plt.cm.magma
                # if coloring should be alpha-level
                # based
                else:
                   _pltargs[k] = None
                   use_rgba = True

    # plot each gene
    for ii in range(n_genes):
        vals = ncnt[:,ii].reshape(-1,)
        if normalize:
            vals = normalize_expression(vals,
                                        c = pseudocount)
        # conduct quantile scaling if specified
        if qscale is not None:
            if qscale > 0 and qscale < 1:
                vals_q = np.quantile(vals,qscale,
                                     interpolation = 'nearest')
                vals[vals > vals_q] = vals_q
            else:
                print('WARNING : {} is not a proper quantile value'.format(qscale),
                      'within range (0,1)')
        # get title 
        title = cnt.columns[ii]
        # split title if specified
        if split_title is not None:
            title = title.split(split_title[0])[int(split_title[1])]


        # set metric information
        if pval:
            metric = r"$-log10(p_{val} + \epsilon$)"
        else:
            metric = r"$t_d$"

        # set title
        ax[ii].set_title('Gene : {} \n'.format(title) + \
                         metric + ': {:0.3f}'.format(rank_values[ii]),
                         fontsize = 15)

        # set plot order to be dependent
        # on expression levels
        # mainly for unstructured data
        high_ordr = np.argsort(vals)
        if use_rgba:
           rgba = np.zeros((ncnt.shape[0],4))
           rgba[:,2] = 1
           rgba[:,3] = vals[high_ordr]
           mx =  rgba[:,3].max()
           mn = rgba[:,3].min()
           rgba[:,3] = (rgba[:,3] - mn) / (mx-mn)
        else:
           rgba = vals[high_ordr]
        # plot
        ax[ii].scatter(crd[:,0][high_ordr],
                       crd[:,1][high_ordr],
                       c = rgba,
                       **_pltargs,
                      )
    # clean all axes
    # for aesthetics
    for ii in range(ncols*nrows):
        clean_axes(ax[ii])

    return (fig,ax)


def timestamp() -> str:
    """generate date-based tag"""
    return re.sub(':|-|\.| |','',
                  str(datetime.datetime.now()))


def banner()->None:
    logo ="                    _   \n"\
          "                  .\ /.  \n"\
          "                 < ~O~ > \n"\
          "┌─┐┌─┐┌─┐┌─┐┬     '/_\\'  \n"\
          "└─┐├┤ ├─┘├─┤│     \ | /  \n"\
          "└─┘└─┘┴  ┴ ┴┴─┘    \|/   "
    v = "Version {} |  see https://github.com/almaan/sepal".format(__version__)
    print( logo + "\n" + v)



def format_timing(t_0 : float,
                  t_end : float,
                  res : pd.DataFrame,
                  method : str,
                  )->Dict[str,Union[str,float,int]]:

    t_tot = t_end - t_0
    n_genes = res.shape[0]

    dct = dict(method = method,
               genes = n_genes,
               time = t_tot,
               )

    return dct
