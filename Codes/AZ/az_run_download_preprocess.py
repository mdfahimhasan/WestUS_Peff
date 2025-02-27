# # importing necessary libraries and functions
from az_download import download_all_datasets
from az_download_openET import download_all_openET_datasets
from az_preprocesses import run_all_preprocessing

# # nodata, Reference rasters
no_data_value = -9999
model_res = 2000  # in m
AZ_shape = '../../Data_main/AZ_files/ref_files/AZ.shp'
AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'
GEE_merging_refraster_large_grids = '../../Data_main/AZ_files/ref_files/AZ_gee_merge_ref_raster.tif'

# # # #  data download args # # # #
gee_data_list = ['Field_capacity', 'Sand_content', 'GRIDMET_RET',  'GRIDMET_max_RH', 'GRIDMET_Precip',
                 'GRIDMET_short_rad', 'DAYMET_sun_hr', 'DEM', 'Rainy_days']

openET_data_list = ['Irrig_crop_OpenET_IrrMapper', 'Irrigation_Frac_IrrMapper']

years = list(range(1984, 2021))  # 1984 to 2023
months = (1, 12)

data_download_dir = '../../Data_main/AZ_files/rasters'

gee_grid_shape_large = '../../Data_main/AZ_files/ref_files/AZ_gee_grid.shp'
gee_grid_shape_for30m_IrrMapper = '../../Data_main/AZ_files/ref_files/AZ_gee_grid_for30m_IrrMapper.shp'
use_cpu_while_multidownloading = 15

skip_download_gee_data = True                           ######
skip_download_OpenET_data = True                        ######

# # # #  data preprocess args # # # #
skip_process_GrowSeason_data = False                     ######
skip_merge_irrig_frac = True                            ######
skip_merge_irrig_ET = True                              ######
skip_irrig_cropland_classification = True               ######
skip_filter_irrig_ET = True                             ######
skip_summing_irrig_cropET_gs = False                     ######
skip_gridmet_precip_processing = False                   ######
skip_gridmet_RET_precessing = False                      ######
skip_prism_processing = False                            ######
skip_processing_slope_data = False                       ######
skip_process_AWC_data = False                            ######
skip_accum_to_water_year_datasets = False                ######
skip_summing_irrig_cropET_water_yr = False               ######
skip_estimate_precip_intensity = False                   ######
skip_estimate_dryness_index = False                      ######
skip_create_P_PET_corr_dataset = False                   ######
skip_process_lake_raster = False                         ######

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
                                 openET_data_list=openET_data_list,
                                 data_download_dir=data_download_dir,
                                 GEE_merging_refraster=GEE_merging_refraster_large_grids,
                                 westUS_refraster=AZ_raster, westUS_shape=AZ_shape,
                                 skip_download_OpenET_data=skip_download_OpenET_data,
                                 use_cpu_while_multidownloading=use_cpu_while_multidownloading)

    run_all_preprocessing(skip_process_GrowSeason_data=skip_process_GrowSeason_data,
                          skip_merging_irrigated_frac=skip_merge_irrig_frac,
                          skip_merging_irrigated_cropET=skip_merge_irrig_ET,
                          skip_classifying_irrigated_cropland=skip_irrig_cropland_classification,
                          skip_filtering_irrigated_cropET=skip_filter_irrig_ET,
                          skip_gridmet_precip_processing=skip_gridmet_precip_processing,
                          skip_gridmet_RET_precessing=skip_gridmet_RET_precessing,
                          skip_summing_irrigated_cropET_gs=skip_summing_irrig_cropET_gs,
                          skip_prism_processing=skip_prism_processing,
                          skip_processing_slope_data=skip_processing_slope_data,
                          skip_process_AWC_data=skip_process_AWC_data,
                          skip_accum_to_water_year_datasets=skip_accum_to_water_year_datasets,
                          skip_summing_irrigated_cropET_water_yr=skip_summing_irrig_cropET_water_yr,
                          skip_estimate_precip_intensity=skip_estimate_precip_intensity,
                          skip_estimate_dryness_index=skip_estimate_dryness_index,
                          skip_create_P_PET_corr_dataset=skip_create_P_PET_corr_dataset,
                          skip_process_lake_raster=skip_process_lake_raster,
                          ref_raster=AZ_raster)

