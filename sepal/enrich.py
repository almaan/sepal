#!/usr/bin/env python3

"""Functional Enrichment Analysis

Analysis module
for functional enrichment
analysis. Uses g:Profiler.

"""

import pandas as pd
import numpy as np

import os.path as osp

from typing import List,Tuple
from gprofiler import GProfiler
from argparse import Namespace as ARGS

from sepal.utils import iprint,wprint,eprint

def query_genes(genes : List[str],
                organism : str,
                dbs : List[str],
                )-> pd.DataFrame:
    
    """FEA for a list of genes

    Will conduct FEA for a set
    of families. Relies on g:Profiler

    Parameters:
    ----------
    genes : List[str]
        list of genes to be subjected
        to FEA
    organism : str
        organism to use in query
    dbs : List[str]
        databases to query against

    Returns:
    -------
    DataFrame with enriched
    pathways for the specified
    genes set

    """

    gp = GProfiler(return_dataframe = True)
    res = gp.profile(organism = organism,
                    query = genes,
                     )
    n_paths = res.shape[0]
    keep = np.array([x  for x in range(n_paths) if \
                     res['source'].values[x] in
                     dbs])
    res_sel = res.iloc[keep,:]
    res_sel = res_sel.loc[res_sel['significant'],:]

    return res_sel


def analyze_families(families : pd.DataFrame,
                     organism : str = 'hsapiens',
                     dbs : List[str] = "GO:BP",
                     split_title : Tuple[str,int] = None,
                     )->pd.DataFrame:
    """FEA for pattern families

    Will conduct FEA for a set
    of families. Relies on g:Profiler

    Parameters:
    ----------
    families : pd.DataFrame
        gene names as index, familiy id in a
        column named family
    organism : str
        organism to use in query
    dbs : List[str]
        databases to query against
    split_title: Tuple[str,int]
        include if profile names should
        be splitted. First element is string
        to split by, second the element
        to keep.

    Returns:
    -------
    Dataframe with all identfied
    pathways and the family they
    are associated with
    """

    uni_fam = np.unique(families['family'].values)
    all_res = pd.DataFrame([])
    for sel_fam in uni_fam:

        genes = families.index.values[families["family"].values == sel_fam]

        if split_title is not None:
            genes = [x.split(split_title[0])[int( split_title[1] )] for x in genes]
        else:
            genes = genes.tolist()

        _res = query_genes(genes,
                           organism=organism,
                           dbs=dbs,
                           )
        _res['family'] = (np.ones(_res.shape[0]) * sel_fam).astype(int)
        all_res = pd.concat((all_res,_res))

    keep_cols = ['family',
                 'native',
                 'name',
                 'p_value',
                 'source',
                 'intersection_size']

    all_res = all_res.loc[:,keep_cols]
    all_res.index = pd.Index(np.arange(all_res.shape[0]) + 1)

    return all_res

def main(args : ARGS,
         )->None:

    """CLI FEA"""

    iprint("Initiating FEA")
    iprint("Using Databases : {}".format(','.join(args.databases)),
           )

    families = pd.read_csv(args.family_index, sep = '\t',
                           header = 0,
                           index_col = 0)

    enr = analyze_families(families,
                           args.organism,
                           args.databases,
                           args.split_title,
                           )

    enr['family'] += args.start_at

    obase = osp.basename(args.family_index)
    obase = obase.replace("family-index",
                          "FEA")

    opth = osp.join(args.out_dir,obase)

    enr.to_csv(opth,
               index = True,
               header = True,
               sep = '\t',
               )

    if args.latex:
        lopth = opth.replace(".tsv","-latex.txt")
        with open(lopth,"w+") as f:
            f.write(enr.to_latex())

    if args.markdown:
        mopth = opth.replace(".tsv","-markdown.md")
        with open(mopth,"w+") as f:
            f.write(enr.to_markdown())

    return None
