#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os.path as osp
from scipy.stats import spearmanr


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
        _tmp = pd.read_csv(vals['file'],sep = '\t',header = 0,index_col = 0)
        _tmp = _tmp[[vals['column']]]
        print(_tmp.head())

        if genes is None:
            genes = _tmp.index
        else:
            genes = genes.intersection(_tmp.index)
        met_list.update({met:_tmp})


    print(genes)
    for k,v in met_list.items():
        met_list[k] = v.loc[genes,:]

    return met_list



#-------------
sum_fun = lambda x: x.sum(axis=0)
var_fun = lambda x: x.var(axis=0)

funs = dict(total_sum = sum_fun,
            variance = var_fun)

cnt_pth = "~/w-projects/spatential/data/mob/Rep11_MOB.st_mat.processed.tsv"
cnt = pd.read_csv(cnt_pth, header = 0, index_col = 0, sep = '\t')

methods = dict(sepal = dict( file = "/home/alma/w-projects/spatential/res/publication/mob/20200407115358366240-top-diffusion-times.tsv",
                               column = "average",
                               ),
               spatialDE = dict(file = "/home/alma/w-projects/spatential/res/publication/spatialDE/spatialDE-mob.tsv",
                                  column = 'qval',
                                  ),
               SPARK = dict(file = "/home/alma/w-projects/spatential/res/publication/spark/spark-mob.tsv",
                              column = 'adjusted_pvalue'
                              ),
               )

mets = process_methods(methods)
res = evaluate_methods(mets,cnt,funs)
out_dir = "/tmp/mob-4"

for _r in res.keys():
    _tmp = pd.DataFrame(res[_r])
    _tmp.index = ['spearman','p-value']
    with open(osp.join(out_dir,_r + "-comp-results.tsv"),"w") as f:
        ostream = _tmp.to_latex()
        ostream = ostream.replace("lrrr","l|c|c|c")
        f.writelines(ostream)

