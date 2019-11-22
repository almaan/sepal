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

    prs.add_argument('-fg','--filter_genes',
                     type = bool,
                     default = True,
                     help = 'output directory')

    prs.add_argument('-ar','--array',
                     type = str,
                     choices = ['visium',
                                '2k',
                                '1k'],
                     required = True,
                     help = 'array type')

    plt_prs = sub_prs.add_parser('plot',)

    plt_prs.add_argument('-sd','--style_dict',
                         required = False,
                         default = None,
                         type = json.loads,
                         help = 'plot style as dict',
                         )

    plt_prs.add_argument('-ss','--side_size',
                         required = False,
                         type = float,
                         help = 'side length in plot',
                         )

    plt_prs.add_argument('-nc','--ncols',
                     type = int,
                     default = 5,
                     help = 'number f columns in plot')

    plt_prs.add_argument('-ng','--n_genes',
                     type = int,
                     default = 20,
                     help = 'number of columns in plot')

    plt_prs.add_argument('-qs','--quantile_scaling',
                         default = None,
                         type = float,
                         help = 'quantile to use for quantile scaling',
                         )

    pth = "/home/alma/Documents/PhD/papers/STSC/data/fetal_heart/st/w9/D2_CN31_st_data.processed.tsv"
# TODO : remove and uncomment on launch 
#    args = prs.parse_args()
    args = prs.parse_args(['-c',
                           pth,
                           '-o',
                           '/tmp/',
                           '-ar',
                           '1k',
                           'plot',
                           '-ss',
                            '4.2',
                           '-sd',
                           '{"s" : 20, "edgecolor":"black"}',
                           ],
                          )
    #TODO: laplacian function must be adjusted as well
    if args.array == '1k':
        setup = dict(radius = 1,
                     rotate = None,
                     n_neighbours = 4,
                     )

    elif args.array == '2k':
        setup = dict(radius = np.sqrt(2),
                     rotate = 45,
                     n_neighbours = 4,
                     )
    elif args.array == 'visium':
        print("ERROR : method is not yet",
              "implemented for {} arrays".format(args.array)
              )
        sys.exit(-1)



    for sample,cpth in enumerate(args.count_files):
        cnt = ut.read_file(cpth)
        cnt = ut.filter_genes(cnt)
        cd = ut.CountData(cnt,
                          **setup, 
                          )

        times = ut.propagate(cd)

        out_df = pd.DataFrame([str(x) for x in times],
                              index = cd.cnt.columns,
                              columns = ['diff-time'],
                              )
        out_df_pth = osp.join(args.out_dir,'diffusion-times.tsv')

        out_df.to_csv(out_df_pth,
                      sep = '\t',
                      header = True,
                      index = True,
                      )

        if 'plot' in args.modules:
            fig,ax = ut.visualize_genes(cd,
                                        times,
                                        ncols = args.ncols,
                                        side_size = args.side_size,
                                        n_genes = args.n_genes,
                                        qscale = args.quantile_scaling,
                                        )
            fig.savefig(osp.join(args.out_dir,'diffusion-times.svg'))

main()


