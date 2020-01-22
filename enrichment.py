#!/usr/bin/env python3

from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

import numpy as np
import pandas as pd

from os import mkdir
import os.path as osp

from typing import List

pandas2ri.activate()

def enrichment_analysis(res: pd.DataFrame,
                        dbs : list = ["GO_Biological_Process_2018"],
                        alpha : float = 0.05) -> List[pd.DataFrame]:

    try:
        enrichR = importr("enrichR")
    except RRuntimeError:
        print("[ERROR] : Please install R enrichR package")
        return []

    base = importr("base")

    av_dbs = enrichR.listEnrichrDbs()["libraryName"].values

    dbs = list(filter(lambda x : x in av_dbs, dbs))

    if len(dbs) < 1:
        print("[ERROR] : None of the specified databases are available")
        return []

    cluster_ids = np.unique(res.values[:,0])
    cluster_ids = np.sort(cluster_ids[cluster_ids >= 0])
    enriched  = dict() 

    for k,db in enumerate(dbs):
        enriched.update({db:{}})
        for cluster in cluster_ids:

            pos = res.values[:,0] == cluster
            if pos.sum() < 2:
                enriched[db].update({cluster:pd.DataFrame([])})
                continue

            genes = ro.StrVector(res.index.values[pos])

            enrres_r = enrichR.enrichr(genes,base.c(db))
            enrres_df = enrres_r.rx2(k+1)

            enrres_df = enrres_df.iloc[enrres_df['Adjusted.P.value'].values < alpha,:]

            enriched[db].update({cluster:enrres_df})

    return enriched 

def save_enrihment_results(enr_res : dict,
                           out_dir : str,
                           )->None:


    out_dir_enr = osp.join(out_dir,"enrichment")
    if not osp.exists(out_dir_enr):
        mkdir(out_dir_enr)

    for db_name,res_list in enr_res.items():
        for cluster,res in res_list.items():
            bname = 'enr-db-' + db_name + '-cl-' + str(cluster) + '.tsv'
            res.to_csv(osp.join(out_dir_enr,bname),
                       sep = '\t',
                       header = True,
                       index = True)

    return None
