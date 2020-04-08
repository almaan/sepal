#!/usr/bin/env python3

import argparse as arp
import os.path as osp
from os import mkdir

import json

import numpy as np
import pandas as pd

#import enrichment as enr
import utils as ut
import models as m
from datasets import RawData



class SettingVars:
    """default variables"""
    def __init__(self,):
        self.SEL_COLUMN = 'average'
        self.PVAL_EPS = 1e-273

def enrich(args):
    return None

def family(times_all,cd,sampletag,args):
    st = SettingVars()
    sort_genes = np.argsort(times_all[st.SEL_COLUMN].values)[::-1]

    if args.n_base_genes is None:
        args.n_base_genes = np.min((args.n_genes*2,
                                    times_all.shape[0]))
    else:
        args.n_base_genes = np.min((args.n_base_genes,
                                    times_all.shape[0]))


    cluster_genes = times_all.index.values[sort_genes]

    args.threshold = np.clip(args.threshold,0,1)
    cluster_labels,repr_patterns = ut.cluster_data(cd.cnt.loc[:,cluster_genes].values,
                                    n_base = args.n_base_genes,
                                    n_projs = args.n_genes,
                                    threshold = args.threshold,
                                    )

    clusters = pd.DataFrame(cluster_labels,
                            index = cluster_genes[0:args.n_genes],
                            columns = ['family'],
                            )

    out_cl_pth = osp.join(args.out_dir,args.out_dir,'-'.join([sampletag,
                                                            'family',
                                                            'index.tsv']
                                                            ))
    clusters.to_csv(out_cl_pth,
                    sep = '\t',
                    header = True,
                    index = True,
                    )

    if args.plot:

        reprviz,_ = ut.visualize_representative(repr_patterns,
                                                crd = cd.real_crd,
                                                ncols = args.n_cols,
                                                pltargs = args.style_dict,
                                                )

        reproname = osp.join(args.out_dir,''.join([sampletag,
                                                    '-representative.png',
                                                    ],
                                                    ))
        reprviz.savefig(reproname)

        clusterviz = ut.visualize_clusters(cd.cnt.loc[:,cluster_genes[0:args.n_genes]].values,
                                            genes = cluster_genes[0:args.n_genes],
                                            crd = cd.real_crd,
                                            labels = cluster_labels,
                                            ncols = args.n_cols,
                                            pltargs = args.style_dict,
                                            split_title = args.split_title,
                                            )

        for cl in range(len(clusterviz)):
            clustoname = osp.join(args.out_dir,''.join([sampletag,
                                                            '-family-',
                                                                str(cl),
                                                            '.png'],
                                                                ))
            clusterviz[cl][0].savefig(clustoname)

    return None



def topgenes(times_all,cd,sampletag,args):

    st = SettingVars()

    sort_genes = np.argsort(times_all[st.SEL_COLUMN].values)[::-1]

    if args.pval:
        times_all[st.SEL_COLUMN] = -np.log10(times_all[st.SEL_COLUMN].values.flatten() + \
            st.PVAL_EPS)

    if args.n_genes is None:
        args.n_genes= ut.get_inflection_point(times_all[st.SEL_COLUMN].values[sort_genes])

    sel_genes = sort_genes[0:int(args.n_genes)]
    top_genes = times_all.index.values[sel_genes]
    top_times = times_all[st.SEL_COLUMN][sel_genes]

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

    fig.savefig(osp.join(args.out_dir,'-'.join([sampletag,
                                               'top_genes',
                                                'diffusion-times.png']
                                                 )))
    return None

def main(args):
    st = SettingVars()

    sampletag = '.'.join(osp.basename(args.count_data).split('.')[0:-1])

    if not osp.isdir(args.out_dir):
        mkdir(args.out_dir)



    if args.module in ['topgenes','family']:
        times_all = pd.read_csv(args.results,
                                sep = '\t',
                                header = 0,
                                index_col = 0)

        Data = {"1k": m.ST1K,
                "2k": m.ST2K,
                "visium" : m.VisiumData,
                "unstructured": m.UnstructuredData,
                }



        cdata = RawData(args.count_data,
                        args.transpose)

        cd = Data[args.array](cdata,
                              eps = 0.2,
                              normalize = False,
                              )

        if args.module == 'topgenes':
            topgenes(times_all,cd,sampletag,args)
        elif args.module == 'family':
            family(times_all,cd,sampletag,args)
    elif args.module == 'enrich':
        enrich(args)
    else:
        print(["Not a valid module"])


if __name__ == '__main__':
    main()
