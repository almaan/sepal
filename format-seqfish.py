#!/usr/bin/env python3

import pandas as pd
import numpy as np

import sys
import os.path as osp

#cnt_pth = sys.argv[1]
#mta_pth = sys.argv[2]

cnt_pth = "/home/alma/w-projects/spatential/data/SeqFish/seqfish_count_data.csv" 
mta_pth = "/home/alma/w-projects/spatential/data/SeqFish/seqfish_meta_data.csv"
out_dir = "/tmp/"

mta = pd.read_csv(mta_pth, sep = ',',header = 0, index_col = 0)
cnt = pd.read_csv(cnt_pth, sep = ',',header = 0, index_col = 0)

cnt = cnt.T
cnt.columns = pd.Index([x.replace("'","") for x in cnt.columns])

mta.index = mta.index.astype(str)
cnt.index = cnt.index.astype(str)

inter = mta.index.intersection(cnt.index)

cnt = cnt.loc[inter,:]
mta = mta.loc[inter,:]

crd = mta[['x','y']].values.astype(str)
nidx = pd.Index(['x'.join(crd[x,:]) for x in range(crd.shape[0])])

cnt.index = nidx

opth = osp.join(out_dir,osp.basename(cnt_pth).replace('.csv','.tsv'))
cnt.to_csv(opth,
           sep = '\t',
           header = True,
           index = True)
