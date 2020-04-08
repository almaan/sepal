#!/usr/bin/env python3
import argparse as arp
import json

def make_parser():

    prs = arp.ArgumentParser()

    parser = arp.ArgumentParser()

    subparsers = parser.add_subparsers(dest = 'command')
    run_parser = subparsers.add_parser("run",
                                       formatter_class=arp.ArgumentDefaultsHelpFormatter)
    analyze_parser = subparsers.add_parser("analyze",
                                        formatter_class=arp.ArgumentDefaultsHelpFormatter)

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

    run_parser.add_argument('-dt','--time_step',
                     type = float,
                     default = 0.001,
                     help = ' '.join(['minimum number of total',
                                      'counts for a gene',
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

    # analyze_parser.add_argument('-al',
    #                 '--analysis',
    #                     nargs = '+',
    #                     default = ['genes'],
    #                     choices = ['genes',
    #                                'cluster',
    #                                'toprank',
    #                                'enrich'])


    # analyze_parser.add_argument('-tg','--top_genes',
    #                  default = None,
    #                  required = False,
    #                   help = ("number of genes"
    #                            " to visualize"
    #                            ))

    # analyze_parser.add_argument('-cg','--cluster_genes',
    #                default = 50,
    #                required = False,
    #                type = int,
    #                help = 'show number of genes',
    #                )

    # analyze_parser.add_argument('-cb','--cluster_base',
    #                default = None,
    #                required = False,
    #                type = int,
    #                help = 'show number of genes',
    #                )

    # analyze_parser.add_argument('-eps','--threshold',
    #                   default = 0.995,
    #                   type = float,
    #                   help = 'threshold in clustering',
    #                   )

    # analyze_parser.add_argument('-cl','--cluster_labels',
    #                default = None,
    #                required = False,
    #                type = str,
    #                help = 'path to cluster labels',
    #                )


    # analyze_parser.add_argument("-dbs","--databases",
    #                  nargs = '+',
    #                  default = ["GO_Biological_Process_2018"],
    #                  help = ('database to use in enrichment'
    #                          ' analysis'))

    # analyze_parser.add_argument("-pv","--pval",
    #                  default = False,
    #                  action = "store_true",
    #                  help = ('values are pvals'))


    # analyze_parser.add_argument('-sd','--style_dict',
    #                 required = False,
    #                 default = None,
    #                 type = json.loads,
    #                 help = 'plot style as dict',
    #                 )

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
                         nargs = '+',
                         default = None,
                         help = 'split title')

    analyze_subparser = analyze_parser.add_subparsers(dest="module")

    topgene_parser = analyze_subparser.add_parser("topgenes",
                                                  formatter_class=arp.ArgumentDefaultsHelpFormatter)
    family_parser = analyze_subparser.add_parser("family",
                                                  formatter_class=arp.ArgumentDefaultsHelpFormatter)
    enrich_parser = analyze_subparser.add_parser("enrich",
                                                  formatter_class=arp.ArgumentDefaultsHelpFormatter)

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

    enrich_parser.add_argument('-cl','--cluster_labels',
                   default = None,
                   required = False,
                   type = str,
                   help = 'path to cluster labels',
                   )


    enrich_parser.add_argument("-dbs","--databases",
                     nargs = '+',
                     default = ["GO_Biological_Process_2018"],
                     help = ('database to use in enrichment'
                             ' analysis'))

    return parser

