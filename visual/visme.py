#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from os import mkdir
import os.path as osp
import sys

import argparse as arp

import yaml


def read_file(f,sep = '\t'):
    return pd.read_csv(f,sep = sep, header = 0, index_col = 0)

def get_crd(idx):
    return np.array([x.replace('X','').split('x') for x in idx.values]).astype(float)

def clear_ax(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    for sp in ax.spines.values():
        sp.set_visible(False)


prs = arp.ArgumentParser()

prs.add_argument("-c","--count_file",required = False)
prs.add_argument("-r","--results",default = None)
prs.add_argument("-st","--style",default = None,type = str)
prs.add_argument("-gn","--genes",default = None,type = str,nargs = '+')
prs.add_argument("-nt","--ntop_genes",default = None,type = int)
prs.add_argument("-tr","--transpose",default = False,action = "store_true")
prs.add_argument("-ext","--file_format",default = "svg")
prs.add_argument("-o","--out_dir",default = "/tmp")
prs.add_argument("-ms","--marker_size",default = None,type = int)
prs.add_argument("-nn","--no_normalize",default = False,action = "store_true")
prs.add_argument("-rx","--rotate_x",default = False,action = "store_true")
prs.add_argument("-ry","--rotate_y",default = False,action = "store_true")
prs.add_argument("-ir","--include_time",default = False, action = "store_true")
prs.add_argument("-sg","--split_names", nargs = 2,default = None)
prs.add_argument("-int","--use_integer",default = False,action = "store_true")

args = prs.parse_args()

if args.no_normalize:
    trans = lambda x : x
else:
    trans = lambda x: np.log2(x + 2)

if not osp.isdir(args.out_dir):
    mkdir(args.out_dir)

cnt = read_file(args.count_file)
if args.results is not None:
    res = read_file(args.results)
sel_col = "average"

if args.transpose:
    cnt = cnt.T

if args.split_names is not None:
    new_names = [x.split(args.split_names[0])[int(args.split_names[1])] for \
                 x in cnt.columns]

    cnt.columns = pd.Index(new_names)

    new_names = [x.split(args.split_names[0])[int(args.split_names[1])] for x\
                 in res.index]
    res.index = pd.Index(new_names)

crd = get_crd(cnt.index)

if args.use_integer:
    crd = crd.round(0).astype(int)

if args.rotate_x:
    crd = crd[:,[1,0]]

if args.style is not None:
    with open(args.style,"r+") as fopen:
        style = yaml.load(fopen)
        if "style_dict" in style.keys():
            if args.marker_size is not None:
                style['style_dict']['s'] = args.marker_size
            if "cmap" in style['style_dict']:
                style["style_dict"]['cmap'] = eval(style["style_dict"]['cmap'])
        if "fig_dict" not in style.keys():
            side_size = 7
            fontsize = 20
        else:
            side_size = style['fig_dict'].get('side',7)
            fontsize = style['fig_dict'].get('fontsize',20)


if (args.ntop_genes is not None) and \
   (args.results is not None):
    idxs = np.argsort(res[sel_col].values)[::-1][0:args.ntop_genes]
    genes = res.index.values[idxs]

elif args.genes is not None:
    if osp.isfile(args.genes[0]):
        with open(args.genes[0]) as fopen:
            genes = fopen.readlines()
            genes = [x.replace('\n','') for x in genes]
    else:
        genes = args.genes
else:
    sys.exit(-1)

presentgenes = np.array([g.lower() for g in cnt.columns.values])

for _gene in genes:
    if _gene.lower() not in presentgenes:
        continue
    else:
        gpos = np.argmax(_gene.lower() == presentgenes)
        gene = cnt.columns.values[gpos]

    print("Visualizing : {}".format(gene))

    if args.results is not None:
        time = float(res[sel_col].loc[gene])
        title = gene + " | Diffusion Time : {:0.4f}".format(time)
    else:
        title = gene

    figsize = (side_size,side_size)
    fig, ax = plt.subplots(1,1,figsize = figsize)
    ax.scatter(crd[:,0],
            crd[:,1],
            c = trans(cnt[gene].values),
            **style["style_dict"])
    ax.set_title(title,fontsize=fontsize)
    ax.set_aspect("equal")
    clear_ax(ax)

    if args.rotate_y:
        ax.invert_yaxis()

    # fig.tight_layout()
    if args.include_time:
        extra = "{:0.4f}-".format(time)
    else:
        extra = ""
    out_pth = osp.join(args.out_dir,extra + gene + "." + args.file_format)

    fig.savefig(out_pth,transparent = True, dpi = 600)

plt.close("all")
