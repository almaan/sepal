#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

import argparse as arp
from typing import List


np.random.seed(1337)


def ablate_set(cnt : pd.DataFrame,
               crd : np.ndarray,
               n_shuffle : np.array,
               )->List[pd.DataFrame]:

    n_seeds = cnt.shape[1]
    n_spots = cnt.shape[0]

    abl_sets = []

    for ii in range(n_seeds):
        vals = cnt.values[:,ii]

        abl = pd.DataFrame(np.zeros((crd.shape[0],
                                     n_shuffle.shape[0])),
                        index = cnt.index,
                        columns = ["Shuffled : " + str(x) for x in n_shuffle],
                        )
        all_idx = np.arange(n_spots)

        for k,shuf in enumerate(n_shuffle):
            expr = vals.copy()
            idx = np.random.choice(all_idx,
                                   size = shuf,
                                   replace=False)

            expr[idx] = expr[np.random.permutation(idx)]
            abl.iloc[:,k] = expr

        abl_sets.append(abl)

    return abl_sets


def main()->None:

    prs = arp.ArgumentParser()
    aa = prs.add_argument

    aa("-s","--seed",
       required =True,
       type = str,
       help ="seeding set",
       )

    aa("-o","--out_dir",
       required =True,
       type = str,
       help ="seeding set",
       )

    aa("-ns","--shuffle",
       nargs = "+",
       type = int,
       default = None,
       )

    args = prs.parse_args()

    cnt = pd.read_csv(args.seed,
                      sep = '\t',
                      header = 0,
                      index_col = 0,
                      )

    crd = np.array([x.split('x') for x in \
                    cnt.index.values]).astype(float)

    original_pos = [x for x in cnt.columns.values if\
                    x.split('-')[1] == 'O']

    cnt = cnt.loc[:,pd.Index(original_pos)]

    if args.shuffle is None:
        n_shuffle = np.array([0,100,500,900])
    else:
        n_shuffle = np.array(args.shuffle)

    n_shuffle = np.clip(n_shuffle,
                        0,
                        cnt.shape[0])

    ablated = ablate_set(cnt,
                         crd,
                         n_shuffle,
                         )

    for k,abl in enumerate(ablated):
        abl.to_csv(osp.join(args.out_dir,
                            "ablated-" + \
                            str(k) + \
                            ".tsv",
                            ),
                   sep = '\t',
                   header = True,
                   index = True)

if __name__ == "__main__":
    main()
