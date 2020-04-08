#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os.path as osp

class RawData:

    def __init__(self,
                 pth : str,
                 transpose : bool = False,
                 ) -> None:

        self.supported_ext = ["h5ad",
                              "csv",
                              "tsv"]

        self._cnt = None
        self._crd = {'pixel':None,
                     'array' : None,
                     }

        self.original_data = dict(file_path = pth)

        self._use_crd = 'pixel'

        self.ext = self.get_ext(pth)

        if self.ext is not None:
            self.read_data(pth,
                           self.ext,
                           transpose,
                           )

    def get_ext(self,
                fname : str,
                )->str:

        bname = osp.basename(fname)
        splt = bname.split(".")
        ext = (splt[-1] if not splt[-1]  == 'gz' else splt[-2])
        ext = ext.lower()

        if ext not in self.supported_ext:
            print("[WARNING] : {} is not a supported file type")
            ext = None

        return ext

    def read_data(self,
                  fname : str,
                  ext : str,
                  transpose : bool = False,
                  )-> None:

        if ext in ['tsv','csv']:
            sep = dict(tsv = '\t',
                       csv = ',')

            tmp = pd.read_csv(fname,
                              sep = sep[ext],
                              header = 0,
                              index_col = 0,
                              )

            if transpose:
                tmp  = tmp.T

            self._crd['pixel'] = np.array([[float(x.replace('X','')) for \
                                        x in y.split('x') ] for\
                                        y in tmp.index])

            self._crd['array'] = self._crd['pixel']

            self._cnt = tmp

        elif ext == 'h5ad':
            import anndata as aD

            tmp = aD.read_h5ad(fname)

            n_spots,n_feats = tmp.shape

            if 'x' in tmp.obs.keys() and 'y' in tmp.obs.keys():

                self._crd['pixel'] = np.zeros((n_spots,2))
                self._crd['pixel'][:,0] = tmp.obs['x'].values
                self._crd['pixel'][:,1] = tmp.obs['y'].values

            else:
                self._crd['pixel'] = None

            if '_x' in tmp.obs.keys() and '_y' in tmp.obs.keys():

                self._crd['array'] = np.zeros((n_spots,2))
                self._crd['array'][:,0] = tmp.obs['_x'].values
                self._crd['array'][:,1] = tmp.obs['_y'].values
            else:
                self._crd = None

            self._cnt = pd.DataFrame(tmp.X,
                                    index = tmp.obs.index,
                                    columns = tmp.var['name'].values,
                                    ) 

        else:
            print("[ERROR] : Something went wrong when loading data")

        self.shape = self._cnt.shape


    @property
    def cnt(self,) -> pd.DataFrame:
        return self._cnt

    @cnt.setter
    def cnt(self,value : pd.DataFrame):

        oRws = self._cnt.index.values
        nRws = value.index.values
        kRws = np.array([np.argmax(x == oRws) for x in nRws])

        for crd_type in ['pixel','array']:
            if self._crd[crd_type] is not None:
                self._crd[crd_type] = self._crd[crd_type][kRws,:]

        self._cnt = value
        self.shape = self._cnt.shape

    @property
    def use_crd(self)->str:
        return self._use_crd

    @use_crd.setter
    def use_crd(self,
                crd_type : str, 
                ) -> None:

        if crd_type in ['pixel','array']:
            self._use_crd = crd_type
        else:
            print("[ERROR] : {} is not a supported coordinate type\n".format(crd_type),
                  "defaulting to {} coordinates instead".format(self.use_crd))

    @property
    def crd(self,):
        return self._crd[self.use_crd]

    def __repr__(self,):
        txt = []
        txt.append("RawData object")
        txt.append("\t> loaded from {}".format(self.original_data['file_path']))
        txt.append("\t> using {} coordinates".format(self.use_crd))

        txt = '\n'.join(txt)

        return txt

