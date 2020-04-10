#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os.path as osp
import os
from scipy.stats import spearmanr,pearsonr


def match(sort_vals,
          cnt,
          fun):

    # sorted vals; one columns

    cnt_vals = fun(cnt)
    cnt_vals = pd.DataFrame(cnt_vals,
                            index = cnt.columns,
                            columns = ['x']
                            )

    inter = sort_vals.index.intersection(cnt_vals.index)
    cnt_vals = cnt_vals.loc[inter,:]
    sort_vals = sort_vals.loc[inter,:]

    return (sort_vals.values.flatten(),
            cnt_vals.values.flatten())




def evaluate_methods(mets,cnt,funs):
    corr_list = {}
    for name,fun in funs.items():
        corr_list.update({name:{}})
        for met,vals in mets.items():
            X,Y = match(vals,cnt,fun)
            corr = spearmanr(X,Y)
            corr_list[name].update({met:corr})

    return corr_list

def process_methods(methods):
    genes = None
    met_list = {}
    for met,vals in methods.items():
        _tmp = pd.read_csv(vals['file'],
                           sep = vals['sep'],
                           header = 0,
                           index_col = 0)

        if 'genes' in vals.keys():
            _tmp.index = _tmp[vals['genes']].values
        _tmp = _tmp[[vals['column']]]
        if genes is None:
            genes = _tmp.index
        else:
            genes = genes.intersection(_tmp.index)
        met_list.update({met:_tmp})

    for k,v in met_list.items():
        met_list[k] = v.loc[genes,:]

    return met_list



#-------------
sum_fun = lambda x: x.sum(axis=0)
var_fun = lambda x: x.var(axis=0)

funs = dict(total_sum = sum_fun,
            variance = var_fun)

cnt_pth = "../data/real/mob.tsv.gz"
cnt = pd.read_csv(cnt_pth, header = 0, index_col = 0, sep = '\t')

methods = dict(sepal = dict( file = "../res/mob/20200407115358366240-top-diffusion-times.tsv",
                               column = "average",
                             sep = '\t',
                               ),
               SpatialDE = dict(file = "SpatialDE-MOB_final_results.csv",
                                  column = 'qval',
                                  sep = ',',
                                   genes = 'g',
                                  ),
               SPARK = dict(file = "SPARK-mob.tsv",
                              column = 'combined_pvalue',
                            sep = '\t',
                              ),
               )

mets = process_methods(methods)
res = evaluate_methods(mets,cnt,funs)
out_dir = os.getcwd()


for _r in res.keys():
    _tmp = pd.DataFrame(res[_r])
    _tmp.index = ['spearman ($\\rho$)','p-value']
    print("Correlation with : {}".format(_r))
    print(_tmp,end="\n\n")
    with open(osp.join(out_dir,_r + "-comp-results.tsv"),"w") as f:
        ostream = _tmp.to_latex()
        ostream = ostream.replace("lrrr","l|c|c|c")
        f.writelines(ostream)

