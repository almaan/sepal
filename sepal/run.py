#!/usr/bin/env python3

"""CLI for Diffusion Simulation

Will simulated diffusion in a specified
tissue and provide diffusion times
for all included genes. Results
are saved in tsv files.

This is a submodule of the main
CLI.


"""

import os
import sys
import json
import os.path as osp
import argparse as arp


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sepal.utils as ut
import sepal.models as m
from sepal.datasets import RawData
from argparse import Namespace as ARGS

from sepal.utils import iprint,eprint,wprint
from typing import Mapping, Dict,Tuple,Type


def main(args : ARGS,
         )->None:
    """Run simulation

    Will compute diffusion times
    and save as tsv files

    Parameters:
    ----------
    args : ARGS
        parsed arguments from argparse


    """

    # set tag of run
    analysistag = ut.timestamp()

    # load data
    print("[INFO] : will analyze samples: ")
    for k,s in enumerate(args.count_files):
        print("\t [{}] : {}".format(k + 1,s))

    Data : Mapping[str,Type[m.CountData]] = {"1k": m.ST1K,
                                            "2k": m.ST2K,
                                            "visium" : m.VisiumData,
                                            "unstructured": m.UnstructuredData,
                                            }


    times_all = []
    sampletags = []
    # analyze each provided sample
    for sample,cpth in enumerate(args.count_files):

        sampletags.append('.'.join(osp.basename(cpth).split('.')[0:-1]))
        cdata = RawData(cpth,
                        args.transpose)

        # filter data
        if any([args.min_occurance > 0,args.min_counts > 0]):
            iprint("Removing Genes With : "\
                  " TOTAL_EXPR < {} | OCCURANCE < {}".format(args.min_counts,
                                                            args.min_occurance))
            cdata.cnt = ut.filter_genes(cdata.cnt,
                                        min_occur = args.min_occurance,
                                        min_expr = args.min_counts)

        # get dimensions
        n_spots,n_genes = cdata.shape

        print("SAMPLE {} | GENES : {} | SPOTS : {}".format(sample + 1,
                                                           n_genes,
                                                           n_spots))
        # convert to CountData object
        cd = Data[args.array](cdata,
                              eps = 0.2,
                              normalize = False)

        np.random.seed(1337)
        # propagate system in time
        times = m.propagate(cd,
                            dt = args.time_step)

        # save results 
        times_df = pd.DataFrame([str(x) for x in times],
                              index = cd.cnt.columns,
                              columns = [sampletags[-1]],
                              )

        times_all.append(times_df)

    # Min max normalize diffusion times
    times_all = pd.concat(times_all, axis = 1, join = 'inner').astype(float)
    
    mn = times_all.values.min(axis = 0).reshape(1,-1)
    mx = times_all.values.max(axis = 0).reshape(1,-1)

    times_all['average'] = ((times_all.values - mn) / (mx - mn)).mean(axis = 1)

    # save results
    if not osp.exists(args.out_dir):
        os.mkdir(args.out_dir)

    out_df_pth = osp.join(args.out_dir,'-'.join([analysistag,
                                                 'top',
                                                 'diffusion-times.tsv']
                                                 ))

    times_all.to_csv(out_df_pth,
                     sep = '\t',
                     header = True,
                     index = True,
                     )

if __name__ == '__main__':
    main()

