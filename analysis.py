#!/usr/bin/env python3

import argparse as arp
import os.path as osp

import json

import numpy as np
import pandas as pd

import enrichment as enr
import utils as ut
import models as m



def main():

    prs = arp.ArgumentParser()
    sub_prs = prs.add_subparsers(dest = 'modules',)

    prs.add_argument('-c','--count_data',
                     help = 'count files',
                     )

    prs.add_argument('-r','--results',
                     help = 'output directory')


    prs.add_argument('-o','--out_dir',
                     required = True,
                     help = 'output directory')

    prs.add_argument('-ar','--array',
                     type = str,
                     choices = ['visium',
                                '2k',
                                '1k',
                                'unstructured',
                                ],
                     help = 'array type')

    

    
    # gene_prs = sub_prs.add_parser('genes')

    prs.add_argument('-al',
                    '--analysis',
                        nargs = '+',
                        default = ['genes'],
                        choices = ['genes',
                                   'cluster',
                                   'cdf',
                                   'toprank',
                                   'enrich'])


    prs.add_argument('-tg','--top_genes',
                     default = 50,
                     required = False,
                      help = ("number of genes"
                               " to visualize"
                               ))

    prs.add_argument('-cg','--cluster_genes',
                   default = 50,
                   required = False,
                   type = int,
                   help = 'show number of genes',
                   )

    prs.add_argument('-eps','--threshold',
                      default = 0.995,
                      type = float,
                      help = 'threshold in clustering',
                      )

    prs.add_argument('-cl','--cluster_labels',
                   default = None,
                   required = False,
                   type = str,
                   help = 'path to cluster labels',
                   )


    prs.add_argument("-dbs","--databases",
                     nargs = '+',
                     default = ["GO_Biological_Process_2018"],
                     help = ('database to use in enrichment'
                             ' analysis'))


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


    plt_prs.add_argument('-qs','--quantile_scaling',
                         default = None,
                         type = float,
                         help = 'quantile to use for quantile scaling',
                         )



    args = prs.parse_args()

    if args.results is not None:

        times_all = pd.read_csv(args.results,
                                sep = '\t',
                                header = 0,
                                index_col = 0)

        sampletag = '.'.join(osp.basename(args.results).split('.')[0:-1])

        sampletag = sampletag.split('-')[0]

        sel_column = 'average'
        sort_genes = np.argsort(times_all[sel_column].values)[::-1]

    requires_counts = ['cluster',
                       'genes',
                       'toprank']

    if any( [x in args.analysis for x in requires_counts]):

        Data = {"1k": m.ST1K,
                "2k": m.ST2K,
                "visium" : m.VisiumData,
                "unstructured": m.UnstructuredData,
                }

        cnt = ut.read_file(args.count_data)

        cd = Data[args.array](cnt,eps = 0.2,normalize = False)



        sampletag = '.'.join(osp.basename(args.count_data).split('.')[0:-1])

    if 'genes' in args.analysis:
        if args.modules is not None and 'plot' in args.modules:

            sel_genes = sort_genes[0:int(args.top_genes)]
            top_genes = times_all.index.values[sel_genes]
            top_times = times_all[sel_column][sel_genes]

            fig,ax = ut.visualize_genes(cd.cnt.loc[:,top_genes],
                                        cd.real_crd,
                                        top_times,
                                        ncols = args.n_cols,
                                        side_size = args.side_size,
                                        qscale = args.quantile_scaling,
                                        pltargs = args.style_dict,
                                        )

            fig.savefig(osp.join(args.out_dir,'-'.join([sampletag,
                                                        'top_genes',
                                                        'diffusion-times.png']
                                                        )))


    if 'cluster' in args.analysis:

        sel_genes = sort_genes[0:int(args.cluster_genes)]
        cluster_genes = times_all.index.values[sel_genes]

        args.threshold = np.clip(args.threshold,0,1)
        cluster_labels = ut.cluster_data(cd.cnt.loc[:,cluster_genes].values,
                                         threshold = args.threshold,
                                        )

        clusters = pd.DataFrame(cluster_labels,
                                index = cluster_genes,
                                columns = ['cluster'],
                                )

        out_cl_pth = osp.join(args.out_dir,args.out_dir,'-'.join([sampletag,
                                                                'cluster',
                                                                'index.tsv']
                                                                ))
        clusters.to_csv(out_cl_pth,
                        sep = '\t',
                        header = True,
                        index = True,
                        )

        if args.modules is not None and 'plot' in args.modules:
            clusterviz = ut.visualize_clusters(cd.cnt.loc[:,cluster_genes].values,
                                               genes = cluster_genes,
                                               crd = cd.real_crd,
                                               labels = cluster_labels,
                                               ncols = args.n_cols,
                                               pltargs = args.style_dict,
                                               )

            for cl in range(len(clusterviz)):
                clustoname = osp.join(args.out_dir,''.join([sampletag,
                                                            '-cluster-',
                                                                str(cl),
                                                            '.png'],
                                                                ))
                clusterviz[cl][0].savefig(clustoname)

    if 'enrich' in args.analysis:
        if args.cluster_labels is not None:
            clusters= ut.read_file(args.cluster_labels)
        elif 'cluster' in analysis:
            pass 
        else:
            print(("[ERROR] : Provide a path"
                " to the file containing the cluster labels"
                ))
            sys.exit(-1)

        enrichment_results = enr.enrichment_analysis(clusters,
                                                     dbs = args.databases)

        enr.save_enrihment_results(enrichment_results,
                                    args.out_dir)

    if 'toprank' in args.analysis and len(args.analysis) == 1:
        ut.toprank(cnt = cd.cnt,
                   diff = times_all[sel_column])

    if "cdf" in args.analysis:
        cdffig, cdfax = ut.visualize_cdf(diff = times_all[sel_column].values)
        cdfname = osp.join(args.out_dir,''.join([sampletag,
                                                '-cdf',
                                                '.png'],
                                                    ))
        cdffig.savefig(cdfname)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        print("[INFO] : Terminated by user")
