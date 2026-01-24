pkgs <- c(
  "boot","data.table","DescTools","emmeans","evd","extRemes",
  "future","future.apply","geosphere","ggplot2","lmtest","lubridate",
  "MASS","readr","readxl","reshape2","sandwich","scales","stats",
  "stringr","terra","TideHarmonics","tidyr","tidyverse","viridis","xtable"
)

to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

