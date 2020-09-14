#!/usr/bin/env python3

"""CLI for Analysis of Results

Provides modules for
downstream analysis such
as generation of pattern families,
enrichment analysis, and visualization
of top genes

"""


import argparse as arp
import os.path as osp
from os import mkdir
import sys

import json

import numpy as np
import pandas as pd

import sepal.utils as ut
from sepal.utils import iprint,eprint,wprint,VARS

from sepal.enrich import main as fea
from sepal.family import main as families
import sepal.models as m
from sepal.datasets import RawData

from argparse import Namespace as ARGS

def topgenes(times_all,
             cd,
             sampletag,
             args,
             ):

    iprint("Visualizing top profiles")
    if args.pval:
        times_all[VARS.SEL_COLUMN] = -np.log10(times_all[VARS.SEL_COLUMN]\
                                               .values.flatten() + \
                                               VARS.PVAL_EPS)

    # sort genes by rank metric
    sort_genes = np.argsort(times_all[VARS.SEL_COLUMN]\
                            .values)[::-1]

    # automatically determine number
    # of top genes to use
    if args.n_genes is None:
        args.n_genes= ut.get_knee_point(times_all[VARS.SEL_COLUMN]\
                                        .values[sort_genes],
                                        sigma = args.sigma,
                                        )

    # create visualizations
    sel_genes = sort_genes[0:int(args.n_genes)]
    top_genes = times_all.index[sel_genes]
    top_genes = top_genes.intersection(cd.cnt.columns)
    top_times = times_all.loc[top_genes,VARS.SEL_COLUMN]

    
    fig,ax = ut.plot_profiles(cd.cnt.loc[:,top_genes],
                              cd.real_crd,
                              top_times,
                              ncols = args.n_cols,
                              side_size = args.side_size,
                              qscale = args.quantile_scaling,
                              pltargs = args.style_dict,
                              split_title = args.split_title,
                              pval = args.pval,
                              )
    # save visualization
    fig.savefig(osp.join(args.out_dir,'-'.join([sampletag,
                                               'top_genes',
                                                'diffusion-times.png']
                                                 )))
    return None

def main(args : ARGS,
         )->None:
    """Analyze results

    Provides means for
    inspection and downstream
    analysis of the spatial
    patterns identified

    Parameters:
    ----------
    args : ARGS
       parsed arguments from argparse


    """

    # set tag for sample
    sampletag = '.'.join(osp.basename(args.count_data).split('.')[0:-1])

    # create output directory
    # if non-existing
    if not osp.isdir(args.out_dir):
        mkdir(args.out_dir)


    # execute specified analysis
    if args.module in ['inspect','family']:
        iprint("will analyze : \n >CNT :"\
               "{} \n >RANKS : {}".format(args.count_data,
                                         args.results))
        if args.array is None:
            eprint("Please specify array type")
            sys.exit(-1)

        # load results
        times_all = pd.read_csv(args.results,
                                sep = '\t',
                                header = 0,
                                index_col = 0)

        # set array type
        Data = {"1k": m.ST1K,
                "2k": m.ST2K,
                "visium" : m.VisiumData,
                "unstructured": m.UnstructuredData,
                }

        # read count data
        cdata = RawData(args.count_data,
                        args.transpose,
                        only_include = times_all.index,
                        )


        # convert count data to compatible
        # CountData object
        cd = Data[args.array](cdata,
                              eps = 0.2,
                              normalize = False,
                              )

        # visualizes top profiles
        if args.module == 'inspect':
            topgenes(times_all,
                     cd,
                     sampletag,
                     args)
        # generates families 
        elif args.module == 'family':
            families(times_all,
                     cd,
                     sampletag,
                     args)
    # conducts enrichment analysis
    elif args.module == 'fea':
        fea(args)
    else:
        print(["Not a valid module"])


if __name__ == '__main__':
    main()
