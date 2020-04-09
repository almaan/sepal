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

import json

import numpy as np
import pandas as pd

import utils as ut
from utils import iprint,eprint,wprint
import models as m
from datasets import RawData

from argparse import Namespace as ARGS



class VARS:
    """default variables"""
    SEL_COLUMN = 'average'
    PVAL_EPS = 1e-273

def enrich(args : ARGS,
           )->None:
    return None

def family(times_all :pd.DataFrame,
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
    family_labels,repr_patterns = ut.get_families(cd.cnt.loc[:,use_genes].values,
                                    n_base = args.n_base_genes,
                                    n_sort = args.n_genes,
                                    threshold = args.threshold,
                                    )
    # save family assortment
    families = pd.DataFrame(family_labels,
                            index = use_genes[0:args.n_genes],
                            columns = ['family'],
                            )
    
    out_fl_pth = osp.join(args.out_dir,args.out_dir,'-'.join([sampletag,
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

        reprviz,_ = ut.plot_representative(repr_patterns,
                                           crd = cd.real_crd,
                                           ncols = args.n_cols,
                                           pltargs = args.style_dict,
                                           )

        reproname = osp.join(args.out_dir,''.join([sampletag,
                                                    '-representative.png',
                                                    ],
                                                    ))
        reprviz.savefig(reproname)

        family_plots = ut.plot_families(cd.cnt.loc[:,use_genes[0:args.n_genes]].values,
                                        genes = use_genes[0:args.n_genes],
                                        crd = cd.real_crd,
                                        labels = family_labels,
                                        ncols = args.n_cols,
                                        pltargs = args.style_dict,
                                        split_title = args.split_title,
                                        )

        for fl in range(len(family_plots)):
            famoname = osp.join(args.out_dir,''.join([sampletag,
                                                      '-family-',
                                                       str(fl),
                                                       '.png'],
                                                        ))
            family_plots[fl][0].savefig(famoname)

    return None



def topgenes(times_all,
             cd,
             sampletag,
             args):

    iprint("Visualizing top profiles")
    if args.pval:
        times_all[VARS.SEL_COLUMN] = -np.log10(times_all[VARS.SEL_COLUMN].values.flatten() + \
            VARS.PVAL_EPS)

    # sort genes by rank metric
    sort_genes = np.argsort(times_all[VARS.SEL_COLUMN].values)[::-1]

    # automatically determine number
    # of top genes to use
    if args.n_genes is None:
        args.n_genes= ut.get_inflection_point(times_all[VARS.SEL_COLUMN].values[sort_genes])

    # create visualizations
    sel_genes = sort_genes[0:int(args.n_genes)]
    top_genes = times_all.index.values[sel_genes]
    top_times = times_all[VARS.SEL_COLUMN][sel_genes]

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
                        args.transpose)

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
            family(times_all,cd,sampletag,args)
    # conducts enrichment analysis
    elif args.module == 'enrich':
        enrich(args)
    else:
        print(["Not a valid module"])


if __name__ == '__main__':
    main()
