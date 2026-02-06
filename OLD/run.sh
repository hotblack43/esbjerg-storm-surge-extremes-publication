# create the on-the-hour-data
rm OUTPUT/ONTHEHOUR/*
./RunChunck.scr GESLA_data_checker_2.Rmd
# fit Harmonic tidal model and generate MSL and annaual max residuals
rm OUTPUT/ANNUALS/*
./RunChunck.scr ftide_residuals_3.Rmd
# test S vs NS GEV
#uv run S_vs_NS_tester_6_bayes_or_resampling.py
# collect and filter the significant ones
./RunChunck.scr consider_collected_S_vs_NS_results_2.Rmd
# make table
./RunChunck.scr make_publication_table_4.Rmd
# make figures
uv run make_map_lon_lat_tidegauge_stations_14_satellite_zoom.py
