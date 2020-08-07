#!/usr/bin/env python3

##-------------------------------------------------------------
##  SpatialDE Data Analysis  
##-------------------------------------------------------------

## From : https://github.com/Teichlab/SpatialDE/README.md
## Downloaded : 2020-08-07

## Modified to time the analysis

import pandas as pd
import time
import yaml

import NaiveDE
import SpatialDE

import sepal.utils as ut
import argparse as arp
import os.path as osp

prs = arp.ArgumentParser()
aa = prs.add_argument

aa("-c","--count_data",required = True)
aa("-m","--meta_data",required = True)
aa("-o","--out_dir",required = True)
aa("-t","--tag",default = None)
aa("-z","--timeit",default = False, action = "store_true")


args = prs.parse_args()

if args.tag is None:
    tag = ""
else:
    tag = args.tag + "-"

counts = pd.read_csv(args.count_data,
                     index_col=0)

counts = counts.T[counts.sum(0) >= 3].T  # Filter practically unobserved genes

sample_info = pd.read_csv(args.meta_data,
                          index_col=0)

counts = counts.loc[sample_info.index]  # Align count matrix with metadata table

norm_expr = NaiveDE.stabilize(counts.T).T
resid_expr = NaiveDE.regress_out(sample_info,
                                 norm_expr.T,
                                 'np.log(total_counts)').T

# sample_resid_expr = resid_expr.sample(n=100,
#                                       axis=1,
#                                       random_state=1)

X = sample_info[['x', 'y']]

t_0 = time.time()
results = SpatialDE.run(X, resid_expr)
t_end = time.time()

timing_dct = ut.format_timing(t_0,t_end,results,method ="SpatialDE")

results.to_csv(osp.join(args.out_dir,tag + ".tsv"),
                  sep = '\t',
                  header = True,
                  index = True)

if args.timeit:
    with open(osp.join(args.out_dir,tag +"timing.yaml"),"w") as f:
        _ = yaml.dump(timing_dct,f,default_flow_style = False)
