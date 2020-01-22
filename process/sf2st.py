#!/usr/bin/env python3

import numpy as np
import pandas as pd


import os.path as osp


cnt_pth = "/home/alma/w-projects/spatential/data/SeqFish/seqfish_count_data.csv"
mta_pth = "/home/alma/w-projects/spatential/data/SeqFish/seqfish_meta_data.csv"


cnt = pd.read_csv(cnt_pth,
                  sep = ',',
                  header = 0,
                  index_col = 0)

mta = pd.read_csv(mta_pth,
                  sep = ',',
                  header = 0,
                  index_col = 0)

inter = mta.index.intersection(cnt.columns)

cnt = cnt.loc[:,inter]
mta = mat.loc[inter,:]
