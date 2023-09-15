# # importing necessary libraries and functions
from Codes.data_download_process.download_process import download_all_datasets, run_all_preprocessing
from Codes.hydro_process_dist.sw_dist import compile_USGS_WaterUse_data

# # Reference raster
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'

# # Data download
skip_download_gee_data = True
skip_download_ssebop_data = True

gee_data_list = ['GPW_Pop', 'MODIS_NDWI']
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
         2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

months = (1, 12)
download_all_datasets(year_list=years, month_range=months, gee_data_list=gee_data_list,
                      skip_download_gee_data=skip_download_gee_data,
                      skip_download_ssebop_data=skip_download_ssebop_data)

# # Run preprocessing steps
skip_cdl_cropland_processing = True
cdl_already_masked = True
try_maximum_occurrence_approach_for_cdl = True
skip_ssebop_processing = True
skip_irrigatedCrop_processing = True
skip_prism_processing = True
run_all_preprocessing(skip_cdl_cropland_processing=skip_cdl_cropland_processing, cdl_already_masked=cdl_already_masked,
                      try_maximum_occurrence_approach_for_cdl=try_maximum_occurrence_approach_for_cdl,
                      skip_ssebop_processing=skip_ssebop_processing,
                      skip_irrigatedCrop_processing=skip_irrigatedCrop_processing,
                      skip_prism_processing=skip_prism_processing)

# # Compiling USGS county-scale GW data
skip_compiling = False  # # Set to False to compile USGS water use data
years = [2005, 2010, 2015]

WaterUse_usgs_csv = compile_USGS_WaterUse_data\
    (yearlist=years, usgs_data_dir='../Data_main/USGS_water_use_data',
     county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county_projected.shp',
     irrigated_data_dir='../Data_main/Raster_data/Irrigated_agriculture/WestUS_Processed_irrigated_lands',
     county_id_data='../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif',
     output_dir='../Data_main/USGS_water_use_data',
     skip_compiling=skip_compiling, ref_raster=WestUS_raster)

