ESBJERG2 / PUBLICATION (Linux Only)
Purpose

This pipeline analyzes stationary vs non-stationary storm-surge extremes
using GESLA tide-gauge data. You will produce intermediate outputs and
final results.

Quick Start

Install R and RStudio if you haven’t already. RStudio makes running
.Rmd files easy.

Open RStudio and ensure you are in this folder:

esbjerg-storm-surge-extremes-publication/

Install the required packages by running the following in the RStudio
Console:

source("install_packages.R")


Run the .Rmd files in the correct order as specified in
RUN_ORDER.txt. Open each .Rmd file in RStudio and click "Knit" or run
chunks interactively.

For the map generation step (if part of your workflow), run the Python
script as noted in the RUN_ORDER.txt.

Outputs

Intermediate files are saved in OUTPUT/ folders (e.g., ONTHEHOUR2/,
ANNUALS2/).

Final results appear in RESULTS/ (e.g., collected test results).

Optional Helpers

For auxiliary scripts, tips, or quality checks, see QUICK_START.txt.
