
# # importing necessary libraries and functions
from download import download_all_datasets
from preprocesses import run_all_preprocessing

# # Reference raster
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

# # Data download
skip_download_gee_data = False
skip_download_ssebop_data = True

gee_data_list = \
    ['USDA_CDL'
# 'OpenET_ensemble',
    # 'GRIDMET_Precip'
     #    'MODIS_Day_LST', 'MODIS_NDWI', 'MODIS_LAI', 'MODIS_NDVI',
     # 'TERRACLIMATE_vap_pres', 'GRIDMET_RET', 'GRIDMET_vap_pres_def',  'GRIDMET_max_RH',
     # 'GRIDMET_Precip', 'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
     # 'Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID', 'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID',
     # 'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID', 'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID',
     # 'Field_capacity', 'Bulk_density', 'Organic_carbon_content','Sand_content', 'Clay_content', 'DEM'
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

# # Run preprocessing steps
skip_ssebop_processing = True
skip_prism_processing = True
skip_merging_rainfed_frac = True
skip_merging_rainfed_cropET = True
skip_merging_irrigated_frac = True
skip_merging_irrigated_cropET = True
skip_classifying_irrigated_rainfed_cropland = True
skip_filtering_irrigated_rainfed_cropET = True
skip_processing_slope_data = True

run_all_preprocessing(skip_ssebop_processing=skip_ssebop_processing,
                      skip_prism_processing=skip_prism_processing,
                      skip_merging_rainfed_frac=skip_merging_rainfed_frac,
                      skip_merging_rainfed_cropET=skip_merging_rainfed_cropET,
                      skip_merging_irrigated_frac=skip_merging_irrigated_frac,
                      skip_merging_irrigated_cropET=skip_merging_irrigated_cropET,
                      skip_classifying_irrigated_rainfed_cropland=skip_classifying_irrigated_rainfed_cropland,
                      skip_filtering_irrigated_rainfed_cropET=skip_filtering_irrigated_rainfed_cropET,
                      skip_processing_slope_data=skip_processing_slope_data,
                      ref_raster=WestUS_raster)
