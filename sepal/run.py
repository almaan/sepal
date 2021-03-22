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
import time
import os.path as osp
import yaml
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
                                        min_expr = args.min_counts,
                                        max_zero_percentage = args.max_zero_fraction,
                                        filter_spurious = (not args.keep_spurious),
                                        )

        # get dimensions
        n_spots,n_genes = cdata.shape

        print("SAMPLE {} | GENES : {} | SPOTS : {}".format(sample + 1,
                                                           n_genes,
                                                           n_spots))
        # convert to CountData object
        # TODO: remove time here
        t_0 = time.time()
        cd = Data[args.array](cdata,
                              eps = 0.2,
                              normalize = False)

        np.random.seed(1337)
        # propagate system in time

        #t_0 = time.time()
        times = m.propagate(cd,
                            dt = args.time_step,
                            num_workers = args.num_workers,
                            diffusion_rate = args.diffusion_rate,
                            thrs = args.threshold,
                            pseudocount = args.pseudocount,
                            )
        t_end = time.time()

        timing_res = ut.format_timing(t_0,
                                      t_end,
                                      times,
                                      method = "sepal",
                                      )

        times.columns = pd.Index([sampletags[-1]])

        if args.timeit:
            if not osp.exists(args.out_dir):
                os.mkdir(args.out_dir)
            with open(osp.join(args.out_dir,
                               sampletags[-1] +"-timing.yaml"),"w") as f:

                _ = yaml.dump(timing_res,f,default_flow_style = False)


        times_all.append(times)

    # Min max normalize diffusion times
    times_all = pd.concat(times_all,
                          axis = 1,
                          join = 'inner').astype(float)

    mn = times_all.values.min(axis = 0).reshape(1,-1)
    mx = times_all.values.max(axis = 0).reshape(1,-1)

    avs = ut.safe_div(times_all.values - mn, mx - mn)
    avs = avs.mean(axis = 1)
    times_all['average'] = avs

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

