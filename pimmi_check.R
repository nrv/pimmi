rm(list=ls())
setwd("/home/nherve/Travail/Workspaces/pimmi")
library(data.table)
library(tidyverse)

data <- as.data.frame(fread(file="/rex/local/otmedia/tweetimages/20201109/pimmi/pq16.test.mining_sub.csv", check.names=FALSE, head=TRUE, stringsAsFactors=FALSE, quote = "\"", sep=",", dec="."))
data$avg_score <- data$score / data$nb

plot_data <- data %>% filter(nb > 4)
plot_data <- data %>% filter(query_image_id == "000257.jpg")

hist(plot_data$nb, breaks=100)
hist(plot_data$score, breaks=100)
hist(plot_data$avg_score, breaks=500)
