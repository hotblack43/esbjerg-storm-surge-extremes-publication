# set up package manager 'uv'
#
# curl -Ls https://astral.sh/uv/install.sh | sh
# uv init
#
mkdir DATA/
ls ~/GESLA4/DATA/* > DATA/GESLA_FILENAMES_HOME2.txt
# ONCE do this:
uv run find_coastal_gesla.py 
cp ./COASTAL_55Y.txt DATA/GESLA_FILENAMES_HOME2.txt
uv run station_info.py     
uv run GESLA_index_maker_2.py 
#
#
#--------------------------------------------------------------------
# assuming a list of good tide-gauge series has been established, proceed with
uv run GESLA_data_checker_4_FAST_RDSFIX.py --jobs 12
# in Rstudio run ftide_residuals_5_SH_winteralso.Rmd
uv run qc_annuals2_2.py --indir OUTPUT/ANNUALS2 --outdir OUTPUT/ANNUALS2_QC --plot 1 > qc_output.txt
uv run S_vs_NS_tester_9_resampling.py
# to get qq an dpp and PIT diagrams: uv run S_vs_NS_tester_10_resampling_GOF_with_pooledPIT.py
uv run consider_collected_S_vs_NS_results_4.py
uv run make_publication_table_4.py
#uv run make_map_lon_lat_tidegauge_stations_15_satellite_zoom.py
uv run make_map_TRIANGLES_6.py
#--------------------------------------------------------------------
# Other, auxiliary codes exist - e.g. for making bathymetric station maps
# and interactive visual quality checking.
#--------------------------------------------------------------------
# To produce the boxplots and the heatmap, do
# uv run seabed_slope_depth_fast.py --column depth_band_median_m
# uv run summarise_best_ring.py
# uv run seabed_slope_depth_fast.py --column slope_median
# uv run summarise_best_ring.py
#--------------------------------------------------------------------
# PIT scores
# uv run parametric_bootstrap_PIT_scores.py  --B 1200 --make_plots
# uv run caseA_table_and_beta_hist.py

