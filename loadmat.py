#!/usr/bin/env python3

# import modules, define some functions for loading, saving and processing a gene-barcode matrix
import collections
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import h5py

import os.path as osp

import argparse as arp


FeatureBCMatrix = collections.namedtuple('FeatureBCMatrix', ['feature_ids', 'feature_names', 'barcodes', 'matrix'])

def get_matrix_from_h5(filename):
    with h5py.File(filename) as f:
        if u'version' in f.attrs:
            if f.attrs['version'] > 2:
                raise ValueError('Matrix HDF5 file format version (%d) is an newer version that is not supported by this function.' % version)
        else:
            raise ValueError('Matrix HDF5 file format version (%d) is an older version that is not supported by this function.' % version)
        
        feature_ids = [x.decode('ascii', 'ignore') for x in f['matrix']['features']['id']]
        feature_names = [x.decode('ascii', 'ignore') for x in f['matrix']['features']['name']]        
        barcodes = list(f['matrix']['barcodes'][:])
        matrix = sp_sparse.csc_matrix((f['matrix']['data'], f['matrix']['indices'], f['matrix']['indptr']), shape=f['matrix']['shape'])
        return FeatureBCMatrix(feature_ids, feature_names, barcodes, matrix)

def get_expression(fbm, gene_name):
    try:
        gene_index = feature_bc_matrix.feature_names.index(gene_name)
    except ValueError:
        raise Exception("%s was not found in list of gene names." % gene_name)
    return fbm.matrix[gene_index, :].toarray().squeeze()

def main():

    prs = arp.ArgumentParser()

    prs.add_argument("-c",'--count_file',
                     type = str,
                     required = True,
                     help = 'count file',
                     )
    
    prs.add_argument("-tp",'--tissue_positions',
                     type = str,
                     required = True,
                     help = 'tissue_positions',
                     )

    prs.add_argument("-o",'--out_dir',
                     type = str,
                     required = True,
                     help = 'output directory',
                     )

    prs.add_argument("-px",'--pixel_crd',
                     default = False,
                     action = 'store_true',
                     help = 'output directory',
                     )

    args = prs.parse_args()


    filtered_matrix_h5 = args.count_file 
    feature_bc_matrix = get_matrix_from_h5(filtered_matrix_h5)

    pos = pd.read_csv(args.tissue_positions,
                      sep = ',',
                      header = None,
                      index_col = 0,
                      )
    pos.columns = ['under_tissue','x','y','pix_x','pix_y']

    index = feature_bc_matrix.barcodes
    index = [x.decode('UTF-8') for x in index]
    if args.pixel_crd:
        sel = ['pix_x','pix_y']
    else:
        sel = ['x','y']

    index = pos.loc[index,sel].values.astype(str)
    index = ['x'.join(x) for x in index]
    
    mat = feature_bc_matrix.matrix.todense().T
    mat = pd.DataFrame(mat,
                       index = index,
                       columns = feature_bc_matrix.feature_names)
    
    opth = osp.join(args.out_dir,osp.basename(filtered_matrix_h5).replace('.h5','.tsv.gz'))
    mat.to_csv(opth, sep = '\t', header = True, index = True, compression = 'gzip')


if __name__ == '__main__':
    main()
