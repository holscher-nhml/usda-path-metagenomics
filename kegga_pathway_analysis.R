# Imports & Setup
library(limma)
library(glue)
library(dplyr)

setwd("./kegga") # set to the "kegga" folder

# Load gene universe
universe = read.table("./universe.csv", row.names = 1, skip = 1)
universe = rownames(universe)

# Load DGEs
for (food in c("Almond", "Broccoli", "Walnut")) {
  file = glue("dge_{food}.csv")
  table = read.table(file, sep = ",", header = TRUE)
  dges = table
  
  var = glue("dge.{food}")
  assign(var, dges)
}

# Load + filter KEGG database
genes = getGeneKEGGLinks("ko")
pathways = getKEGGPathwayNames("ko", remove = TRUE)

genes = genes %>% filter(!grepl('map', PathwayID))
pathways = pathways %>% filter(!grepl('map', PathwayID))

# Run kegga
for (food in c("Almond", "Broccoli", "Walnut")) {
  dge = glue("dge.{food}")
  results = kegga(
    get(dge)$genes,
    geneid = get(dge)$genes,
    species.KEGG = "ko",
    universe = universe,
    restrict.universe = TRUE,
    gene.pathway = genes,
    pathway.names = pathways
  )
  
  # filter and sort on p value
  results = results %>% filter(P.DE < 0.05)
  results = results[order(results$P.DE), ]
  
  # save results
  write.csv(results,
            glue("../results/{food}_top_pathways.csv"),
            row.names = TRUE)
}
