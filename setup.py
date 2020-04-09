#!/usr/bin/env python3

from setuptools import setup
import os
import sys

setup(name='Sepal',
            version='1.0.0',
            description='Identification of Spatial Patterns By Diffusion Modelling',
            url='http://github.com/almaan/sepal',
            author='Alma Andersson',
            author_email='almaan@kth.se',
            license='MIT',
            packages=['sepal'],
            python_requires='>3.0.0',
            install_requires=['numba>=0.46.0',
                               'numpy',
                               'pandas',
                               'matplotlib',
                               'joblib',
                               'scipy',
                            ],
            extras_require={"h5ad":  ["anndata"],
                            "unstructured": ["lap"],
                            "progress": ["tqdm",],
                            "families":["scikit-learn"],
                            "enrichment":["gprofiler-official"],
                            },
            entry_points={'console_scripts': ['sepal = sepal.__main__:main',
                                             ]
                         },
            zip_safe=False)
