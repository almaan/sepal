#!/usr/bin/Rscript

library(SPARK)

#cnt_pth <- "/home/alma/w-projects/spatential/data/fake-patterns/20200127161150201659-pattern-count-matrix.tsv"
cnt_pth <-"~/w-projects/spatential/data/mob/Rep11_MOB.st_mat.processed.tsv"
cnt_pth <- "/home/alma/w-projects/spatential/data/fake-patterns/ablation-sets/ablated-9.tsv"

cnt <- read.table(cnt_pth, sep = '\t',header = 1, row.names = 1)
rn <- rownames(cnt)
info <- cbind.data.frame(x = as.numeric(sapply(strsplit(rn, split = "x"), "[", 1)), 
                         y = as.numeric(sapply(strsplit(rn, split = "x"), "[", 2)))
rownames(info) <- rn

spark <- CreateSPARKObject(counts = t(cnt), location=info[,1:2],percentage = 0.01,min_total_counts = 1)

spark@lib_size <- apply(spark@counts,2,sum)

spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size, 
                  num_core = 8, verbose = T, fit.maxiter = 500)
spark <- spark.test(spark, check_positive = T, verbose = T)


ordr <- order(as.numeric(spark@res_mtest[,c("adjusted_pvalue")]))
## top20 <- head(spark@res_mtest[ordr,c("combined_pvalue","adjusted_pvalue")],n=20)
out <- spark@res_mtest[ordr,]

opth = "/home/alma/w-projects/spatential/res/publication/spark/spark-mob.tsv"
write.table(out,opth, col.names =  T, row.names = T, sep = '\t')

