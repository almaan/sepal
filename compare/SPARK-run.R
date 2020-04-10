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

## read the raw counts
## EDIT : changed path to data
counts <- read.table("./spark-raw_data/Rep11_MOB_count_matrix-1.tsv.gz", check.names = F)
rn <- rownames(counts)
info <- cbind.data.frame(x = as.numeric(sapply(strsplit(rn, split = "x"), "[", 1)), 
                         y = as.numeric(sapply(strsplit(rn, split = "x"), "[", 2)))
rownames(info) <- rn

spark <- CreateSPARKObject(counts = t(counts), location = info[, 1:2], 
                           percentage = 0.1, min_total_counts = 10)

spark@lib_size <- apply(spark@counts, 2, sum)
spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size, 
                  num_core = 10, verbose = T, fit.maxiter = 500)
spark <- spark.test(spark, check_positive = T, verbose = T)
## EDIT : commented out write Rds

#save(spark, file = "./output/Rep11_MOB_spark.rds")

## EDIT : added write tsv file
write.table(spart@res_mtest, "SPARK-mob.tsv",sep = '\t')

## EDIT : removed subsequent analysis
