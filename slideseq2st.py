#!/usr/bin/env python3

import pandas as pd
import numpy as np

import sys
import os.path as osp

#cnt_pth = sys.argv[1]
#mta_pth = sys.argv[2]
#out_dir = sys.argv[3]

cnt_pth = "/home/alma/w-projects/spatential/data/SlideSeq/MappedDGEForR.csv" 
mta_pth = "/home/alma/w-projects/spatential/data/SlideSeq/BeadLocationsForR.csv"
out_dir = "/tmp/"

mta = pd.read_csv(mta_pth, sep = ',',header = 0, index_col = 0)
cnt = pd.read_csv(cnt_pth, sep = ',',header = 0, index_col = 0)

cnt = cnt.T

mta.index = mta.index.astype(str)
cnt.index = cnt.index.astype(str)

inter = mta.index.intersection(cnt.index)

cnt = cnt.loc[inter,:]
mta = mta.loc[inter,:]

crd = mta[['xcoord','ycoord']].values.astype(str)
nidx = pd.Index(['x'.join(crd[x,:]) for x in range(crd.shape[0])])

cnt.index = nidx

opth = osp.join(out_dir,osp.basename(cnt_pth).replace('.csv','.tsv'))
cnt.to_csv(opth,
           sep = '\t',
           header = True,
           index = True)
