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
                         default = 3.5,
                         help = 'side length in plot',
                         )

    plt_prs.add_argument('-nc','--n_cols',
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

    plt_prs.add_argument('-cl','--cluster',
                         default = False,
                         action = 'store_true',
                         help = 'spaital clustering',
                         )

    plt_prs.add_argument('-eps','--threshold',
                         default = 0.8,
                         type = float,
                         help = 'threshold in clustering',
                         )



    args = prs.parse_args()

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
        setup = dict(radius = np.sqrt(2),
                     rotate = 45,
                     n_neighbours = 4,
                     )
    else:
        print('array type not supported')
        sys.exit(-1)



    for sample,cpth in enumerate(args.count_files):
        sampletag = '.'.join(osp.basename(cpth).split('.')[0:-1])
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

        print("GENES : {} | SPOTS : {}".format(n_genes,n_spots))

        cd = ut.CountData(cnt,
                          **setup, 
                          )

        np.random.seed(1337)
        times = ut.propagate(cd)

        out_df = pd.DataFrame([str(x) for x in times],
                              index = cd.cnt.columns,
                              columns = ['diff-time'],
                              )

        if not osp.exists(args.out_dir):
            os.mkdir(args.out_dir)

        
        out_df_pth = osp.join(args.out_dir,args.out_dir,'-'.join([sampletag,
                                                                  'top',
                                                                  'diffusion-times.tsv']
                                                                 )
                              )

        out_df.to_csv(out_df_pth,
                      sep = '\t',
                      header = True,
                      index = True,
                      )

        if 'plot' in args.modules:
            fig,ax = ut.visualize_genes(cd,
                                        times,
                                        ncols = args.n_cols,
                                        side_size = args.side_size,
                                        n_genes = args.n_genes,
                                        qscale = args.quantile_scaling,
                                        pltargs = args.style_dict,
                                        )

            fig.savefig(osp.join(args.out_dir,'-'.join([sampletag,
                                                        'top',
                                                        'diffusion-times.svg']
                                                       )
                                 )
                        )

            if args.cluster:
                args.threshold = np.clip(args.threshold,0,1)
                eigviz, clusterviz = ut.visualize_clusters(cd,
                                                          times,
                                                          args.n_genes,
                                                          args.n_cols,
                                                          threshold = args.threshold,
                                                          pltargs = args.style_dict,
                                                          )

                patoname = osp.join(args.out_dir,'-'.join([sampletag,
                                                          'patterns',
                                                          'diffusion-times.svg'],
                                                        )
                                  )
                                     

                eigviz[0].savefig(patoname) 

                for cluster in range(len(clusterviz)):
                    clustoname = osp.join(args.out_dir,'-'.join([sampletag,
                                                                'cluster',
                                                                 str(cluster),
                                                                'diffusion-times.svg'],
                                                                 )
                                         )
                    
                    clusterviz[cluster][0].savefig(clustoname) 


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Terminated by user")


