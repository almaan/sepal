#!/usr/bin/env python3


import numpy as np
import pandas as pd
import SpatialDE
import NaiveDE

import sys

cnt_pth = "../data/real/mob.tsv.gz"
counts = pd.read_csv(cnt_pth,
                     header = 0,
                     index_col = 0,
                     sep = '\t')
crd = np.array([x.replace('X','').split('x') for x in counts.index.values]).astype(float)
sample_info = pd.DataFrame(np.hstack((crd,counts.values.sum(axis = 1).reshape(-1,1))),
                           columns = ['x','y','total_counts'],
                           index = counts.index,
                           )
norm_expr = NaiveDE.stabilize(counts.T).T
resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T
results = SpatialDE.run(crd,resid_expr)
print(results.sort_values('qval').head(20)[['g', 'l', 'qval']])
out = results[['qval']]
out.index = results['g'].values
out.to_csv("/home/alma/w-projects/spatential/res/publication/spatialDE/spatialDE-mob.tsv",sep = '\t',header = True,index =True)
