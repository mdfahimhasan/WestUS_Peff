# # importing necessary libraries and functions
from download import download_all_datasets
from download_openET import download_all_openET_datasets
from preprocesses import run_all_preprocessing

# # nodata, Reference rasters
no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'

# # # #  data download args # # # #
gee_data_list = ['Field_capacity', 'Bulk_density', 'Sand_content', 'Clay_content',
                 'GRIDMET_RET', 'GRIDMET_vap_pres_def','GRIDMET_max_RH','GRIDMET_Precip',
                 'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                 'TERRACLIMATE_SR', 'USDA_CDL', 'Tree_cover', 'DEM', 'Rainy_days']

openET_data_list = ['Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID',
                    'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID'
                    'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID',
                    'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID',
                    'OpenET_ensemble', 'OpenET_indiv_models_grow_season']

years = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
months = (1, 12)

gee_grid_shape_large = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'
gee_grid_shape_for30m_IrrMapper = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m_IrrMapper.shp'
gee_grid_shape_for30m_LANID = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m_LANID.shp'
data_download_dir = '../../Data_main/Raster_data'
use_cpu_while_multidownloading = 15

skip_download_gee_data = True                           ######
skip_download_OpenET_data = True                       ######

# # # #  data preprocess args # # # #
skip_process_GrowSeason_data = True                     ######
skip_prism_processing = True                            ######
skip_gridmet_precip_processing = True                   ######
skip_gridmet_RET_precessing = True                      ######
skip_merging_rainfed_frac = True                        ######
skip_merging_rainfed_cropET = True                      ######
skip_merging_irrigated_frac = True                      ######
skip_merging_irrigated_cropET = True                    ######
skip_classifying_irrigated_rainfed_cropland = True      ######
skip_filtering_irrigated_rainfed_cropET = True          ######
skip_summing_irrigated_cropET_gs = True                 ######
skip_summing_rainfed_cropET_gs = True                   ######
skip_summing_rainfed_cropET_water_yr = True             ######
skip_openET_sum = True                                  ######
skip_excess_ET_filter_processing = True                 ######
skip_processing_slope_data = True                       ######
skip_process_AWC_data = True                            ######
skip_peff_training_data_filtering = True                ######
skip_accum_to_water_year_datasets = True                ######
skip_summing_irrigated_cropET_water_yr = True           ######
skip_estimate_runoff_precip_frac = True                 ######
skip_estimate_precip_intensity = True                   ######
skip_estimate_dryness_index = True                      ######
skip_process_ksat_data = True                           ######
skip_process_rel_infil_cap_data = True                  ######
skip_create_P_PET_corr_dataset = True                   ######
skip_estimate_peff_water_yr_frac = True                 ######
skip_lake_raster_creation = False                        #####

# # # #  runs # # # #
if __name__ == '__main__':
    download_all_datasets(year_list=years, month_range=months,
                          grid_shape_large=gee_grid_shape_large,
                          data_download_dir=data_download_dir,
                          gee_data_list=gee_data_list,
                          skip_download_gee_data=skip_download_gee_data,
                          use_cpu_while_multidownloading=use_cpu_while_multidownloading)

    download_all_openET_datasets(year_list=years, month_range=months,
                                 grid_shape_for_2km_ensemble=gee_grid_shape_large,
                                 grid_shape_for30m_irrmapper=gee_grid_shape_for30m_IrrMapper,
                                 grid_shape_for30m_lanid=gee_grid_shape_for30m_LANID,
                                 openET_data_list=openET_data_list,
                                 data_download_dir=data_download_dir,
                                 GEE_merging_refraster=GEE_merging_refraster_large_grids,
                                 westUS_refraster=WestUS_raster, westUS_shape=WestUS_shape,
                                 skip_download_OpenET_data=skip_download_OpenET_data,
                                 use_cpu_while_multidownloading=use_cpu_while_multidownloading)

    run_all_preprocessing(skip_process_GrowSeason_data=skip_process_GrowSeason_data,
                          skip_prism_processing=skip_prism_processing,
                          skip_gridmet_precip_processing=skip_gridmet_precip_processing,
                          skip_gridmet_RET_precessing=skip_gridmet_RET_precessing,
                          skip_merging_rainfed_frac=skip_merging_rainfed_frac,
                          skip_merging_rainfed_cropET=skip_merging_rainfed_cropET,
                          skip_merging_irrigated_frac=skip_merging_irrigated_frac,
                          skip_merging_irrigated_cropET=skip_merging_irrigated_cropET,
                          skip_classifying_irrigated_rainfed_cropland=skip_classifying_irrigated_rainfed_cropland,
                          skip_filtering_irrigated_rainfed_cropET=skip_filtering_irrigated_rainfed_cropET,
                          skip_summing_irrigated_cropET_gs=skip_summing_irrigated_cropET_gs,
                          skip_summing_rainfed_cropET_gs=skip_summing_rainfed_cropET_gs,
                          skip_summing_rainfed_cropET_water_yr=skip_summing_rainfed_cropET_water_yr,
                          skip_sum_openET=skip_openET_sum,
                          skip_excess_ET_filter_processing=skip_excess_ET_filter_processing,
                          skip_processing_slope_data=skip_processing_slope_data,
                          skip_process_AWC_data=skip_process_AWC_data,
                          skip_effective_precip_training_data_filtering=skip_peff_training_data_filtering,
                          skip_accum_to_water_year_datasets=skip_accum_to_water_year_datasets,
                          skip_summing_irrigated_cropET_water_yr=skip_summing_irrigated_cropET_water_yr,
                          skip_estimate_peff_water_yr_frac=skip_estimate_peff_water_yr_frac,
                          skip_estimate_runoff_precip_frac=skip_estimate_runoff_precip_frac,
                          skip_estimate_precip_intensity=skip_estimate_precip_intensity,
                          skip_estimate_dryness_index=skip_estimate_dryness_index,
                          skip_process_ksat_data=skip_process_ksat_data,
                          skip_process_rel_infiltration_capacity_data=skip_process_rel_infil_cap_data,
                          skip_create_P_PET_corr_dataset=skip_create_P_PET_corr_dataset,
                          skip_create_lake_raster=skip_lake_raster_creation,
                          ref_raster=WestUS_raster)

