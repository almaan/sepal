#!/usr/bin/env python3

import os
import sys
import json
import os.path as osp
import argparse as arp


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils as ut
import models as m

def main():

    prs = arp.ArgumentParser()
    sub_prs = prs.add_subparsers(dest = 'modules',
                                 )

    prs.add_argument('-c','--count_files',
                     nargs = '+',
                     required = True,
                     help = 'count files',
                     )
    prs.add_argument('-o','--out_dir',
                     required = True,
                     help = 'output directory')

    prs.add_argument('-t','--transpose',
                     required = False,
                     default = False,
                     action = 'store_true',
                     help = 'output directory')

    prs.add_argument('-mo','--min_occurance',
                     type = int,
                     default = 5,
                     help = ' '.join(['minimum number of spot',
                                      'that gene has to occur within',
                                      ],
                                     )
                     )

    prs.add_argument('-mc','--min_counts',
                     type = int,
                     default = 20,
                     help = ' '.join(['minimum number of total',
                                      'counts for a gene',
                                      ],
                                     )
                     )

    prs.add_argument('-ar','--array',
                     type = str,
                     choices = ['visium',
                                '2k',
                                '1k',
                                'unstructured',
                                ],
                     required = True,
                     help = 'array type')

    analysistag = ut.timestamp()

    print("[INFO] : will analyze samples: ")
    for k,s in enumerate(args.count_files):
        print("\t [{}] : {}".format(k + 1,s))

    Data = {"1k": m.ST1K,
            "2k": m.ST2K,
            "visium" : m.VisiumData,
            "unstructured": m.UnstructuredData,
            }

    times_all = []
    data_list = []
    sampletags = []
    for sample,cpth in enumerate(args.count_files):

        sampletags.append('.'.join(osp.basename(cpth).split('.')[0:-1]))
        cnt = ut.read_file(cpth)
        if args.transpose:
            cnt = cnt.T
        cnt.index = pd.Index([x.lstrip('X') for x in cnt.index])

        if any([args.min_occurance > 0,args.min_counts > 0]):
            print("Removing Genes With : ",
                  "TOTAL_EXPR < {} | OCCURANCE < {}".format(args.min_counts,
                                                            args.min_occurance))
            cnt = ut.filter_genes(cnt,
                                  min_occur = args.min_occurance,
                                  min_expr = args.min_counts)

        n_spots,n_genes = cnt.shape

        print("SAMPLE {} | GENES : {} | SPOTS : {}".format(sample + 1,
                                                           n_genes,
                                                           n_spots))

        cd = Data[args.array](cnt,eps = 0.2,normalize = False)

        np.random.seed(1337)
        times = ut.propagate(cd)


        times_df = pd.DataFrame([str(x) for x in times],
                              index = cd.cnt.columns,
                              columns = [sampletags[-1]],
                              )

        times_all.append(times_df)

        data_list.append(cd)

    
    times_all = pd.concat(times_all, axis = 1, join = 'inner').astype(float)
    
    mn = times_all.values.min(axis = 0).reshape(1,-1)
    mx = times_all.values.max(axis = 0).reshape(1,-1)

    times_all['average'] = ((times_all.values - mn) / (mx - mn)).mean(axis = 1)

    if not osp.exists(args.out_dir):
        os.mkdir(args.out_dir)

    out_df_pth = osp.join(args.out_dir,args.out_dir,'-'.join([analysistag,
                                                                'top',
                                                                'diffusion-times.tsv']
                                                                ))

    times_all.to_csv(out_df_pth,
                        sep = '\t',
                        header = True,
                        index = True,
                        )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Terminated by user")


