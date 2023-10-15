
# # importing necessary libraries and functions
from download import download_all_datasets

# # Reference raster
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'

# # Data download
skip_download_gee_data = False
skip_download_ssebop_data = True

gee_data_list = [
    # 'Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID', 'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID',
    # 'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID', 'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID',
    # 'USDA_CDL', 'MODIS_Day_LST', 'MODIS_NDWI', 'MODIS_LAI',
    # 'Field_capacity', 'Bulk_density', 'Organic_carbon_content','Sand_content', 'Clay_content'
]

years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
          2016, 2017, 2018, 2019, 2020]

months = (1, 12)
gee_grid_shape_large = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'
gee_grid_shape_for30m_IrrMapper = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m_IrrMapper.shp'
gee_grid_shape_for30m_LANID = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m_LANID.shp'
data_download_dir = '../../Data_main/Raster_data'
use_cpu_while_multidownloading = 15
download_all_datasets(year_list=years, month_range=months,
                      grid_shape_large=gee_grid_shape_large, grid_shape_for30m_irrmapper=gee_grid_shape_for30m_IrrMapper,
                      grid_shape_for30m_lanid=gee_grid_shape_for30m_LANID,
                      data_download_dir=data_download_dir,
                      gee_data_list=gee_data_list,
                      skip_download_gee_data=skip_download_gee_data,
                      use_cpu_while_multidownloading=use_cpu_while_multidownloading,
                      skip_download_ssebop_data=skip_download_ssebop_data)

# # # Run preprocessing steps
# skip_cdl_cropland_processing = True
# cdl_already_masked = True
# try_maximum_occurrence_approach_for_cdl = True
# skip_ssebop_processing = True
# skip_irrigatedCrop_processing = True
# skip_prism_processing = True
# skip_OpenET_processing = True
#
#
# run_all_preprocessing(skip_cdl_cropland_processing=skip_cdl_cropland_processing, cdl_already_masked=cdl_already_masked,
#                       try_maximum_occurrence_approach_for_cdl=try_maximum_occurrence_approach_for_cdl,
#                       skip_ssebop_processing=skip_ssebop_processing,
#                       skip_prism_processing=skip_prism_processing)
