#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact as fe

import matplotlib.pyplot as plt
import argparse as arp

def toprank(cnt : pd.DataFrame,
            diff : pd.DataFrame,
            )-> None:

    ginter = diff.index.intersection(cnt.columns)
    cnt = cnt.loc[:,ginter]
    diff = diff.loc[ginter,:]
    cnt_s = cnt.values.sum(axis = 0)

    cont = True

    while cont:

        n_examine = input("# genes from result >> ")
        n_topexpr = input("# top expressed genes >> ")

        if n_examine.isdigit() and  n_topexpr.isdigit():
            n_examine = int(n_examine)
            n_topexpr = int(n_topexpr)
        else:
            cont = False
            break

        cnt_g = cnt.columns.values
        cnt_ordr = np.argsort(cnt_s)[::-1]
        cnt_g = cnt_g[cnt_ordr]

        diff_g = diff.index.values
        diff_ordr = np.argsort(diff['average'].values)[::-1]

        diff_g = diff_g[diff_ordr]

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

