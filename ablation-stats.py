#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os

from sklearn.metrics import ndcg_score

def get_rank(index):
    r = [x.replace(" ","").split(":")[-1] for x in index]
    return np.argsort(np.array(r))[::-1]



pth = "/tmp/ablation"
print()

for ii in range(10):
    
    set_dir = osp.join(pth,"set-" + str(ii))
    f = list(filter( lambda x: '.tsv' in x,os.listdir(set_dir)))[0]
    tmp = pd.read_csv(osp.join(set_dir,f),sep = '\t',header = 0, index_col = 0)
    true_rank = get_rank(tmp.index).reshape(1,-1)
    pred_rank = tmp['average'].values.reshape(1,-1)
    # max_rank = dcg_score(true_rank,true_rank)

    # print(true_rank.flatten())
    # print(pred_rank.flatten())

    print(" Set {} | Score : {:0.5f}".format(ii,ndcg_score(true_rank,pred_rank)))



