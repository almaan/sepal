#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os.path as osp
import sys
from typing import Union,Dict,Tuple

class RawData:
    """
    class to hold raw data,
    is compatible with CountData
    class.

    Currently supported files are:

    - tsv
    - csv
    - h5ad (requires anndata package)

    Count data is expected to be in
    format n_locations x n_genes. If
    not, then use the transpose
    argument.

    Arguments:
    ---------

    pth : str
        path to count data
    transpose : bool
        include to transpose count
        data
    only_include: pd.Index
        list of genes to only include


    """

    def __init__(self,
                 pth : str,
                 transpose : bool = False,
                 only_include : pd.Index = pd.Index([]),
                 ) -> None:

        self.supported_ext = ["h5ad",
                              "csv",
                              "tsv"]

        # prepare object variables
        self._cnt : Union[pd.DataFrame] = pd.DataFrame([])
        self._crd : Dict[str,np.ndarray] = {'pixel':np.ndarray([]),
                                            'array' :np.ndarray([]),
                                            }

        # set original data path
        self.original_data = dict(file_path = pth)

        # specify coordinates to use
        # when plotting
        self._use_crd = 'pixel'

        # get extension of file
        self.ext = self.get_ext(pth)

        # error if not supported
        if self.ext not in self.supported_ext:
            print("[WARNING] : {} is not a supported file type")
            sys.exit(-1)

        else:
            self.read_data(pth,
                           self.ext,
                           transpose,
                           only_include,
                           )

    def get_ext(self,
                pth : str,
                )->Union[str,None]:

        """Get extension of path

        Parameters:
        ----------
        pth :
            filename

        Returns:
        -------

        string with extension of file

        """
    
        bname = osp.basename(pth)
        splt = bname.split(".")
        # adjust for gzipped files
        ext = (splt[-1] if not splt[-1]  == 'gz' else splt[-2])
        # set extension to lowercase
        ext = ext.lower()
        return ext

    def read_data(self,
                  pth : str,
                  ext : str,
                  transpose : bool = False,
                  only_include : pd.Index = pd.Index([]),
                  )-> None:
        """Read provided count data

        Parameters:
        ----------
        pth :str
            filename 
        ext : str
            extension of filename
        transpose: bool
            set as true if count data should
            be transposed.
        only_include: pd.Index
            list of genes to only include

        """

        # if file is tsv or csv
        if ext in ['tsv','csv']:
            sep = dict(tsv = '\t',
                       csv = ',')

            tmp = pd.read_csv(pth,
                              sep = sep[ext],
                              header = 0,
                              index_col = 0,
                             )

            if transpose:
                tmp  = tmp.T

            self._crd['pixel'] = np.array([x.replace('X',"").split('x') for \
                                          x in tmp.index.values]).astype(float)
            # take array coordinates as pixel coordinates
            # needed since only one set of coordinates
            # exists
            self._crd['array'] = self._crd['pixel']

            self._cnt = tmp

        # if file is h5ad
        elif ext == 'h5ad':
            # try to import anndata
            # if fails. Exit
            try:
                import anndata as aD
            except ImportError as e:
                print("[ERROR] : you need to install anndata"\
                      " in order to be able to use h5ad files\n"\
                      " See : \n https://anndata.readthedocs.io/en/latest/\n"\
                      " for more information",
                      )
                sys.exit(-1)

            # read h5ad file
            tmp = aD.read_h5ad(pth)
            tmp.var_names_make_unique()

            n_spots,n_feats = tmp.shape
            # set pixel coordinates as x,y

            if "spatial" in tmp.obsm.keys():
                from scipy.sparse.csr import csr_matrix
                gene_names = tmp.var.index
                if isinstance(tmp.X,csr_matrix):
                    dense = False
                else:
                    dense = True

                self._crd['pixel']= tmp.obsm["spatial"][:,[1,0]]

                if ("array_row" in tmp.obs.keys()) and\
                   ("array_col" in tmp.obs.keys()):
                    self._crd["array"] = np.zeros((n_spots,2))
                    self._crd["array"][:,0] = tmp.obs.array_col.values
                    self._crd["array"][:,1] = tmp.obs.array_row.values
                else:
                    self._crd["array"] = None
            else:
                gene_names = tmp.var['name'].values
                dense = True
                if 'x' in tmp.obs.keys() and 'y' in tmp.obs.keys():

                    self._crd['pixel'] = np.zeros((n_spots,2))
                    self._crd['pixel'][:,0] = tmp.obs['x'].values
                    self._crd['pixel'][:,1] = tmp.obs['y'].values

                else:
                    self._crd['pixel'] = None
                # set array coordinates as _x,_y
                if '_x' in tmp.obs.keys() and '_y' in tmp.obs.keys():

                    self._crd['array'] = np.zeros((n_spots,2))
                    self._crd['array'][:,0] = tmp.obs['_x'].values
                    self._crd['array'][:,1] = tmp.obs['_y'].values
                else:
                    self._crd['array'] = None

            # create data frame to hold count
            # data
            self._cnt = pd.DataFrame((tmp.X if dense else np.array(tmp.X.todense())),
                                     index = tmp.obs.index,
                                     columns = gene_names,
                                    )
        else:
            print("[ERROR] : Something went wrong when loading data")

        if len(only_include) > 0:
            inter = self._cnt.columns.intersection(only_include)
            self._cnt = self._cnt.loc[:,inter]


        self.shape = self._cnt.shape

    @property
    def cnt(self,) -> pd.DataFrame:
        """set count data property"""
        return self._cnt

    @cnt.setter
    def cnt(self,
            value : pd.DataFrame,
            )->None:

        """Update count data

        will make sure to adjust coordinates
        for new data. If new data contains
        capture locations not present
        in current data, those will
        be ignored.

        Parameters:
        ----------
        value : pd.DataFrame
            new values

        """

        # get original rownames
        oRws = self._cnt.index.values
        # get new rownames
        nRws = value.index.values
        # sort new data according to
        # old structure.
        kRws = np.array([np.argmax(x == oRws) for x in nRws])
        # rearrange coordinates
        for crd_type in ['pixel','array']:
            if self._crd[crd_type] is not None:
                self._crd[crd_type] = self._crd[crd_type][kRws,:]

        inter =  value.index.intersection(self._cnt.index)
        self._cnt = value.loc[inter,:]
        self.shape = self._cnt.shape

    @property
    def use_crd(self)->str:
        """active coordinates"""
        return self._use_crd

    @use_crd.setter
    def use_crd(self,
                crd_type : str, 
                ) -> None:
        """Set active coordinates

        Parameters:
        ----------

        crd_tyoe : str
            which coordinates to use.
            choose beteen pixerl or array

        """

        if crd_type in ['pixel','array']:
            self._use_crd = crd_type
        else:
            print("[ERROR] : {} is not a supported coordinate type\n".format(crd_type),
                  "defaulting to {} coordinates instead".format(self.use_crd))

    @property
    def crd(self,):
        """Coordinates

        Returns:
        -------
        will return active coordinates
        """
        return self._crd[self.use_crd]

    def __repr__(self,):
        """Representation
        will print information
        regarding original path
        of data and active coordinates

        """
        txt = []
        txt.append("RawData object")
        txt.append("\t> loaded from {}".format(self.original_data['file_path']))
        txt.append("\t> using {} coordinates".format(self.use_crd))

        txt = '\n'.join(txt)

        return txt
