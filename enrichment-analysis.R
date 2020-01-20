#!/usr/bin/Rsript

library(enrichR)

pth <- "/tmp/vis-kid3/20200118075709075331-cluster-index.tsv"
read_table = ro.r("read_table")
res <- read.table(pth, sep = '\t', header = 1, row.names = 1)

dbs <- c("GO_Biological_Process_2018")
alpha <- 0.05

cluster_ids <- unique(res[,1])
enriched  <- list()
for (cluster in cluster_ids){

  pos <- which(res[,1] == cluster)
  cl_genes <- rownames(res)[pos]
  print(cluster)
  print(cl_genes)
  enrres <- enrichr(cl_genes,dbs)
  for (db in dbs) {
    print(enrres[[db]]$Adjusted.P.value)
    print("--")
    enrres[[ db ]] <- enrres[[db]][which(enrres[[db]]$Adjusted.P.value < alpha),]
    print(dim(enrres[[db]]))
  }

  enriched[[as.character(cluster)]] <- enrres
}
