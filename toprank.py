#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact as fe

import matplotlib.pyplot as plt

# diff_pth = "/tmp/vis-mob/20200110135122016273-top-diffusion-times.tsv"
# cnt_pth = "/home/alma/Documents/PhD/papers/STSC/data/molb/st/science/Rep10_MOB.st_mat.processed.tsv"

n_examine = int( sys.argv[1] )
n_topexpr = int( sys.argv[2] )

diff_pth = sys.argv[3]
cnt_pth = sys.argv[4]

read_file = lambda x : pd.read_csv(x,
                                   sep = '\t',
                                   header = 0,
                                   index_col = 0)

diff = read_file(diff_pth)
cnt = read_file(cnt_pth)

ginter = diff.index.intersection(cnt.columns)
cnt = cnt.loc[:,ginter]
diff = diff.loc[ginter,:]


cnt_g = cnt.columns.values
cnt = cnt.values.sum(axis = 0)
cnt_ordr = np.argsort(cnt)[::-1]
cnt = cnt[cnt_ordr]
cnt_g = cnt_g[cnt_ordr]


diff_g = diff.index.values
diff = diff['average'].values
diff_ordr = np.argsort(diff)[::-1]
# diff_ordr = np.argsort(diff)
diff = diff[diff_ordr]
diff_g = diff_g[diff_ordr]

# n_examine = 10
# n_topexpr = 500
sd = set(diff_g[0:n_examine])
sc = set(cnt_g[0:n_topexpr])

inter = sd.intersection(sc)
ninter = len(inter)

cmat = np.array([[ninter,n_examine -ninter,],
                 [n_topexpr - ninter,
                  cnt.shape[0]-n_examine-n_topexpr+ninter]])

_,pval = fe(cmat)

print()
print("[TOP SPATIAL PATTERNS]: {} genes".format(n_examine))
print("[TOP EXPRESSED] : {} genes".format(n_topexpr))
print("[INTERSECTION] : {} genes ".format(ninter))
print("[P-VALUE] : {} ".format(pval))

