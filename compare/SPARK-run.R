#!/usr/bin/Rscript
##-------------------------------------------------------------
##  Mouse OB Data Analysis  
##
##-------------------------------------------------------------

## From : https://github.com/xzhoulab/SPARK-Analysis/blob/master/analysis/SPARK/mouse_ob.R
## Downloaded : 2020-04-10


rm(list = ls())

# load the R package
library(SPARK)
library(yaml)
library(argparse)

parser <- ArgumentParser()

## EDIT : Added parsing functionanlity
parser$add_argument("-c","--count_data")

parser$add_argument("-o","--out_dir")

## default set to same as original code
parser$add_argument("-mc","--min_counts",
                    type = "integer",
                    default = 10)

## default set to same as original code
parser$add_argument("-p","--percentage",
                    type = "double",
                    default = 0.1)

parser$add_argument("-nc","--num_cores",
                    type = "integer",
                    default = 4)

parser$add_argument("-z","--timer",
                    action = "store_true",
                    default = FALSE)

args <- parser$parse_args()

## read the raw counts
## EDIT : changed path to data
counts <- read.table(args$count_data, check.names = F)
rn <- rownames(counts)
info <- cbind.data.frame(x = as.numeric(sapply(strsplit(rn, split = "x"), "[", 1)), 
                         y = as.numeric(sapply(strsplit(rn, split = "x"), "[", 2)))
rownames(info) <- rn

spark <- CreateSPARKObject(counts = t(counts),
                           location = info[, 1:2],
                           percentage = args$percentage,
                           min_total_counts = args$min_counts)

spark@lib_size <- apply(spark@counts, 2, sum)

t0 <- Sys.time()
spark <- spark.vc(spark,
                  covariates = NULL,
                  lib_size = spark@lib_size,
                  num_core = args$num_cores,
                  verbose = T,
                  fit.maxiter = 500)
t_end <- Sys.time()


spark <- spark.test(spark, check_positive = T, verbose = T)
## EDIT : commented out write Rds

#save(spark, file = "./output/Rep11_MOB_spark.rds")

## EDIT : Define output path
opth <- file.path(args$out_dir,
                  "SPARK-mob.tsv")
## EDIT : added write tsv file
write.table(spark@res_mtest,
            opth,
            sep = '\t')

## EDIT : save timing results
if (args$timer){
  t_tot <- as.numeric(t_end) - as.numeric(t_0)
  n_genes <- dim(spark@res_mtest)[1]
  timer_res <- list(method = "spark",
                    genes = n_genes,
                    time = t_tot
                    )

  wite_yaml(timer_res,
            file.path(args$out_dir,"SPARK-times.yaml"))
}

## EDIT : removed subsequent analysis
