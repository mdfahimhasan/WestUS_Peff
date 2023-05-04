# Data download
from Codes.data_download_process.download_process import download_all_datasets, run_all_preprocessing,\
    compile_USGS_WaterUse_data

# Data download
skip_download_gee_data = False
skip_download_ssebop_data = True

# gee_data_list = ['GPW_Pop']
gee_data_list = ['MODIS_NDWI']
years = [2005, 2010, 2015]
months = (1, 12)
download_all_datasets(year_list=years, month_range=months, gee_data_list=gee_data_list,
                      skip_download_gee_data=skip_download_gee_data,
                      skip_download_ssebop_data=skip_download_ssebop_data)
#
# # Run preprocessing steps
# skip_cdl_processing = True
# skip_ssebop_processing = True
# skip_irrigatedCrop_processing = True
# run_all_preprocessing(skip_cdl_processing=skip_cdl_processing, skip_ssebop_processing=skip_ssebop_processing,
#                       skip_irrigatedCrop_processing=skip_irrigatedCrop_processing)
#
# # Compiling USGS county-scale GW data
# include_years = '*2015*.xlsx'  # # '*201[0-5]*.xlsx'
# skip_compiling = False  # # Set to False to compile USGS GW use data
#
# WaterUse_usgs_csv = compile_USGS_WaterUse_data\
#     (data_dir='../../Data_main/USGS_water_use_data', search_by=include_years,
#      county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county_projected.shp',
#      output_csv='../../Data_main/USGS_water_use_data/WestUS_county_WaterUse.csv',
#      skip_compiling=skip_compiling)
#
