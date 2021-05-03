#!/usr/bin/env python3

"""
Parser for sepal

"""

import argparse as arp
import json

def make_parser():

    prs = arp.ArgumentParser()

    parser = arp.ArgumentParser()

    subparsers = parser.add_subparsers(dest = 'command')
    run_parser = subparsers.add_parser("run",
                                       formatter_class=arp\
                                       .ArgumentDefaultsHelpFormatter)
    analyze_parser = subparsers.add_parser("analyze",
                                        formatter_class=arp\
                                           .ArgumentDefaultsHelpFormatter)

    run_parser.add_argument('-c','--count_files',
                     nargs = '+',
                     required = True,
                     help = 'count files',
                     )
    run_parser.add_argument('-o','--out_dir',
                     required = True,
                     help = 'output directory')

    run_parser.add_argument('-t','--transpose',
                     required = False,
                     default = False,
                     action = 'store_true',
                     help = 'transpose count matrix')

    run_parser.add_argument('-mo','--min_occurance',
                            type = int,
                            default = 5,
                            help = ' '.join(['minimum number of spot',
                                             'that gene has to occur within',
                                             ],
                                            )
                            )

    run_parser.add_argument('-mc','--min_counts',
                            type = int,
                            default = 20,
                            help = ' '.join(['minimum number of total',
                                             'counts for a gene',
                                             ],
                                            )
                            )

    run_parser.add_argument('-mzp','--max_zero_fraction',
                            type = float,
                            default = 1.0,
                            help = ' '.join(['max fraction of spots',
                                             'with zero counts allowed for gene',
                                             ],
                                            )
                            )

    run_parser.add_argument('-ks','--keep_spurious',
                            default = False,
                            action = "store_true",
                            help = ' '.join(['include RP and MT',
                                             'profiles',
                                             ],
                                            )
                            )


    run_parser.add_argument('-dt','--time_step',
                     type = float,
                     default = 0.001,
                     help = ' '.join(['minimum number of total',
                                      'counts for a gene',
                                      ],
                                     )
                     )

    run_parser.add_argument('-eps','--threshold',
                     type = float,
                     default = 1e-8,
                     help = ' '.join(['threshold (eps)',
                                      'to use when assessing',
                                      'convergence',
                                      ],
                                     )
                     )

    run_parser.add_argument('-dr','--diffusion_rate',
                     type = float,
                     default = 1,
                     help = ' '.join(['Diffusion rate (D)',
                                     'to use in simulations',
                                      ],
                                     )
                     )


    run_parser.add_argument('-nw','--num_workers',
                     type = int,
                     default = None,
                     help = ' '.join(['number of workers',
                                      'to use. If no number is',
                                      'provided, the maximum',
                                      'number of available workers',
                                      'will be used.',
                                      ],
                                     )
                     )

    run_parser.add_argument('-ar','--array',
                     type = str,
                     choices = ['visium',
                                '2k',
                                '1k',
                                'unstructured',
                                ],
                     required = True,
                     help = 'array type')

    run_parser.add_argument('-z','--timeit',
                            default = False,
                            action = "store_true",
                            required = False,
                            help = 'time analysis')

    run_parser.add_argument("-ps","--pseudocount",
                            type = float,
                            default = 2.0,
                            help = "pseudocount in normalization",
                            )


    # ----- ANALYSIS ------- #
    analyze_parser.add_argument('-c','--count_data',
                     help = 'count files',
                     )

    analyze_parser.add_argument('-r','--results',
                     help = 'output directory')


    analyze_parser.add_argument('-o','--out_dir',
                     required = True,
                     help = 'output directory')

    analyze_parser.add_argument('-ar','--array',
                     type = str,
                     choices = ['visium',
                                '2k',
                                '1k',
                                'unstructured',
                                ],
                     help = 'array type')

    analyze_parser.add_argument('-tr','--transpose',
                     required = False,
                     default = False,
                     action = 'store_true',
                     help = 'transpose count matrix')

    analyze_parser.add_argument('-rt','--rotate',
                     default = False,
                     action = 'store_true',
                     )

    analyze_parser.add_argument('-ss','--side_size',
                     required = False,
                     type = float,
                     default = 350,
                     help = 'side length in plot',
                     )

    analyze_parser.add_argument('-nc','--n_cols',
                 type = int,
                 default = 5,
                 help = 'number f columns in plot')


    analyze_parser.add_argument('-qs','--quantile_scaling',
                     default = None,
                     type = float,
                     help = 'quantile to use for quantile scaling',
                     )

    analyze_parser.add_argument("-st","--split_title",
                         nargs = 2,
                         default = None,
                         help = 'split title')

    analyze_parser.add_argument("-ps","--pseudocount",
                                type = float,
                                default = 2.0,
                                help = "pseudocount in normalization",
                                )

    analyze_parser.add_argument("-sig","--sigma",
                                default = 1.5,
                                type = float,
                                help = "sensitivity for selection of top genes",
                                )

    analyze_subparser = analyze_parser.add_subparsers(dest="module")

    topgene_parser = analyze_subparser.add_parser("inspect",
                                                  formatter_class=arp\
                                                  .ArgumentDefaultsHelpFormatter)
    family_parser = analyze_subparser.add_parser("family",
                                                  formatter_class=arp\
                                                 .ArgumentDefaultsHelpFormatter)
    enrich_parser = analyze_subparser.add_parser("fea",
                                                  formatter_class=arp\
                                                 .ArgumentDefaultsHelpFormatter)

    topgene_parser.add_argument('-sd','--style_dict',
                                required = False,
                                default = None,
                                type = json.loads,
                                help = 'plot style as dict',
                                )

    topgene_parser.add_argument('-nc','--n_cols',
                                type = int,
                                default = 5,
                                help = 'number f columns in plot')

    topgene_parser.add_argument("-pv","--pval",
                     default = False,
                     action = "store_true",
                     help = ('values are pvals'))

    topgene_parser.add_argument('-ng','--n_genes',
                     default = None,
                     required = False,
                      help = ("number of genes"
                               " to visualize"
                               ))

    family_parser.add_argument('-ng','--n_genes',
                   default = 100,
                   required = False,
                   type = int,
                   help = 'included genes',
                   )

    family_parser.add_argument('-nbg','--n_base_genes',
                   default = None,
                   required = False,
                   type = int,
                   help = 'basis genes',
                   )

    family_parser.add_argument('-eps','--threshold',
                      default = 0.995,
                      type = float,
                      help = 'threshold in clustering',
                      )

    family_parser.add_argument('-p','--plot',
                               default = False,
                               action = 'store_true',
                               help = 'threshold in clustering',
                               )
    family_parser.add_argument('-sd','--style_dict',
                                required = False,
                                default = None,
                                type = json.loads,
                                help = 'plot style as dict',
                                )
    family_parser.add_argument('-nc','--n_cols',
                                type = int,
                                default = 5,
                                help = 'number f columns in plot')


    enrich_parser.add_argument('-fl','--family_index',
                                required = True,
                                type = str,
                                help = 'path to family indices',
                                )

    enrich_parser.add_argument('-or','--organism',
                               required = False,
                               default = "hsapiens",
                               type = str,
                               help = 'organism to query against.'\
                               ' See g:Profiler'\
                               " documentation for supported"\
                               " organisms",
                               )


    enrich_parser.add_argument("-dbs","--databases",
                     nargs = '+',
                     default = ["GO:BP"],
                     help = ('database to use in enrichment'
                             ' analysis'))

    enrich_parser.add_argument("-ltx","--latex",
                                default = False,
                                action = 'store_true',
                                help = "save latex formatted table",
                               )
    enrich_parser.add_argument("-md","--markdown",
                                default = False,
                                action = 'store_true',
                                help = "save markdown formatted table",
                               )
    enrich_parser.add_argument("-sa","--start_at",
                                default = 0,
                                type = int,
                                help = "start family enumeration at",
                               )

    return parser


