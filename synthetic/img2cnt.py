#!/usr/bin/env python3

import os
import os.path as osp

from typing import Union, List, Tuple

import re
import datetime

import numpy as np
import pandas as pd

import PIL.Image as im

import argparse as arp


def timestamp() -> str:
    pattern = ':|-|\.| |'
    sub = ""
    date = str(datetime.datetime.now())
    return re.sub(pattern,
                  sub,
                  date)



def threshold( arr : np.ndarray,
               thrs : float,
               low : float = 0,
               high : float = 1,
               )->np.ndarray:

    narr = np.zeros(arr.shape)
    narr[arr > thrs] = high
    narr[arr <= thrs] = low

    return narr

def eprint( s : str) -> None :
    print("[ERROR] : " + s)

def iprint( s : str) -> None :
    print("[INFO] : " + s)


def make_count_matrix(img_files : list,
                      n_children : int = 3,
                      mult_factors : Union[List,Tuple,np.ndarray] = [0.5,1,2],
                      pattern_av : int  = 8,
                      other_av : int = 2,
                      )->pd.DataFrame:


    sizes = [im.open(x).size\
             for x in img_files]

    is_same_size = all([sizes[0] == x for\
                        x in sizes])

    if not is_same_size:
        eprint("Images need to be"
               " of the exact same size"
               )
        raise Exception

    n_patterns = len(img_files)

    n_genes = n_patterns * (1 + n_children*len(mult_factors))
    g_iter = iter(range(n_genes))
    n_spots = sizes[0][0] * sizes[0][1]

    xx = np.arange(0,sizes[0][0])
    yy = np.arange(0,sizes[0][1])

    x,y = np.meshgrid(xx,yy)
    x = x.flatten()
    y = y.flatten()

    gmat = np.zeros((n_spots,n_genes),dtype = np.float)
    gene_names = []

    np.random.seed(1337)

    for num,img_file in enumerate(img_files):

        img = np.asarray(im.open(img_file).convert("L"))

        img = threshold(img,
                        thrs = 127,
                        low = 0,
                        high = 1 )

        vals = img[x,y[::-1]]

        pattern_pos = (vals == 1)
        other_pos = (vals != 1)

        n_pattern_spot = pattern_pos.sum()
        n_other_spot = other_pos.sum() 

        gene_vals = np.zeros(x.shape[0])
        gene_vals[pattern_pos] = np.random.multinomial(pattern_av * n_pattern_spot,
                                                    np.ones(n_pattern_spot) / n_pattern_spot)

        gene_vals[other_pos] = np.random.multinomial(other_av * n_other_spot,
                                                    np.ones(n_other_spot) / n_other_spot)

        gmat[:,next(g_iter)] = gene_vals
        gene_names.append("P" + str(num) + "-" + "O")

        for child in range(n_children):
            for mult in mult_factors:
                gmat[:,next(g_iter)] = (np.random.permutation(gene_vals) * mult).round()
                gene_names.append("P" + str( num ) + \
                                  "-F" + str(child) + "-T" + str( mult ))


    spot_names = [str( xx ) + 'x' + \
                  str( yy ) for \
                  xx,yy in zip(x,y)]

    gmat = pd.DataFrame(gmat,
                        index = spot_names,
                        columns = gene_names)

    return gmat

def main()->None:

    ofile_name = timestamp()

    prs = arp.ArgumentParser()

    prs.add_argument("-if",
                     "--image_files",
                     nargs = '+',
                     help = ("image files to be used"
                             " should all be in the"
                             " format"
                             ),
                     )

    prs.add_argument("-id",
                     "--image_dir",
                     type = str,
                     help = ("directory containing"
                             " image files"
                             ),
                     )

    prs.add_argument("-o",
                     "--out_dir",
                     type = str,
                     default = None,
                     required = False,
                     help = ("output directory"
                             " image files"
                             ),
                     )

    prs.add_argument("-nf",
                      "--n_children",
                      type = int,
                      default = 3,
                      help = ("number of sections"
                              " based on true patterns"
                              " to generate. n_children"
                              " copies for each multiple"
                              " will be formed."
                              ),
                      )

    prs.add_argument("-m",
                     "--multiples",
                     nargs = '+',
                     type = int,
                     default = [0.5,1,2],
                     help = (" multiples (total counts)"
                             " of each pattern to use"
                             )
                     )

    prs.add_argument("-pa",
                     "--pattern_average",
                     type = int,
                     default = 8,
                     help = (" average value for spots"
                             " that constitute the pattern"
                             )
                     )

    prs.add_argument("-po",
                     "--other_average",
                     type = int,
                     default = 2,
                     help = (" average value for spots"
                             " that DO NOT constitute"
                             "the pattern"
                             )
                     )

    args = prs.parse_args()

    if args.image_dir is not None and \
       args.image_files is not None:
            eprint("Provide either only"
                   " image directory"
                   " or image file"
                   )
            raise Exception

    if args.image_dir is not None:
        img_ext = ['png','jpg','gif','tiff','jpeg']
        img_filter = lambda x : x.split('.')[-1].lower() in img_ext
        img_files = list(filter(img_filter, os.listdir(args.image_dir)))
        img_files = list(map(lambda x: osp.join(args.image_dir,x),img_files))
    else:
        img_files = args.image_files

    if isinstance(args.image_files,str):
        img_files = [img_files]

    count_matrix =  make_count_matrix(img_files = img_files,
                                      n_children = args.n_children,
                                      mult_factors = args.multiples,
                                      pattern_av = args.pattern_average,
                                      other_av = args.other_average,
                                      )

    ofile_name = ofile_name + "-pattern-count-matrix.tsv"
    count_matrix.to_csv(osp.join(args.out_dir,
                                 ofile_name),
                        sep = '\t',
                        header = True,
                        index = True)

if __name__ == "__main__":
    main()
