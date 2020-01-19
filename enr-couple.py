#!/usr/bin/env python3

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib.embedded import RRuntimeError

pandas2ri.activate()

import numpy as np
import pandas as pd


def enrichment_analysis(res,dbs,alpha = 0.05):

    try:
        enrichR = importr("enrichR")
    except RRuntimeError:
        print("[ERROR] : Please install R enrichR package")
        return []

    base = importr("base")

    av_dbs = enrichR.listEnrichrDbs()["libraryName"].values

    dbs = list(filter(lambda x : x in av_dbs, dbs))
    print(dbs)

    if len(dbs) < 1:
        print("[ERROR] : None of the specified databases are available")
        return []

    cluster_ids = np.unique(res.values[:,0])
    enriched  = dict() 

    for k,db in enumerate(dbs):
        enriched.update({db:[]})
        for cluster in cluster_ids:

            pos = res.values[:,0] == cluster
            if pos.sum() < 2:
                enriched[db].append(pd.DataFrame([]))
                continue

            genes = ro.StrVector(res.index.values[pos])

            enrres_r = enrichR.enrichr(genes,base.c(db))
            enrres_df = enrres_r.rx2(k+1)

            enrres_df = enrres_df.iloc[enrres_df['Adjusted.P.value'].values < alpha,:]

            enriched[db].append(enrres_df)

    return enriched 


if __name__ == '__main__':

    pth = "/tmp/vis-bc/20200119164303924935-cluster-index.tsv"
    res = pd.read_csv(pth, sep = '\t', header = 0, index_col = 0)
    dbs = ["GO_Biological_Process_2018"]
    test = enrichment_analysis(res,dbs)

    for ii in test[dbs[0]]:
        print(ii)
