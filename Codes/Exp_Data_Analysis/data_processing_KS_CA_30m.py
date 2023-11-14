import os
import sys
from glob import glob

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, sum_rasters, \
    mosaic_rasters_from_directory, clip_resample_reproject_raster
from Codes.utils.ml_ops import create_train_test_dataframe

# Reference resolution, shape, and rasters
no_data_value = -9999
CA_res = 0.0002694945852560897445  # in deg, ~30 m
KS_res = 0.0002694945852680992351  # in deg, ~30 m

gee_merge_KS_30m_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/30m/gee_merge_refraster_KS_30m.tif'
gee_merge_CA_30m_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/30m/gee_merge_refraster_CA_30m.tif'
KS_ROI_30m_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/30m/KS_30m_refraster.tif'
CA_ROI_30m_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/30m/CA_30m_refraster.tif'

KS_shape = '../../Data_main/Data_Drivers_of_ET/ROIs/For_30m/ROI_KS_30m.shp'
CV_shape = '../../Data_main/Data_Drivers_of_ET/ROIs/For_30m/ROI_CA_30m.shp'


# ****** 30m data preprocessing
def merge_gee_data_patches(data_name, years, input_dir, input_shape,
                           gee_merge_ref_raster, roi_clip_ref_raster,
                           resolution, no_data, month_range=None):
    output_dir = os.path.join(input_dir, 'merged')
    interim_output_dir = os.path.join(output_dir, 'gee_merged')
    makedirs([output_dir, interim_output_dir])

    if month_range is None:  # for yearly datasets
        for year in years:
            print(f'merging {data_name} data from GEE for year {year}...')
            search_by = f'*{year}*.tif'

            merge_name = f'{data_name}_{year}.tif'
            merged_arr, merged_raster = \
                mosaic_rasters_from_directory(input_dir=input_dir, output_dir=interim_output_dir,
                                              raster_name=merge_name,
                                              ref_raster=gee_merge_ref_raster,
                                              search_by=search_by, nodata=no_data)

            clip_resample_reproject_raster(input_raster=merged_raster, input_shape=input_shape,
                                           output_raster_dir=output_dir, clip_and_resample=True,
                                           use_ref_width_height=False, resolution=resolution,
                                           ref_raster=roi_clip_ref_raster)

    else:  # for monthly datasets
        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
        for year in years:
            for month in month_list:
                print(f'merging {data_name} data from GEE for year {year}, month {month}...')
                search_by = f'*{year}_{month}_*.tif'  # the extra '_' is needed for collecting the exact data patches

                merge_name = f'{data_name}_{year}_{month}.tif'
                merged_arr, merged_raster = \
                    mosaic_rasters_from_directory(input_dir=input_dir, output_dir=interim_output_dir,
                                                  raster_name=merge_name,
                                                  ref_raster=gee_merge_ref_raster,
                                                  search_by=search_by, nodata=no_data)

                clip_resample_reproject_raster(input_raster=merged_raster, input_shape=input_shape,
                                               output_raster_dir=output_dir, clip_and_resample=True,
                                               use_ref_width_height=False, resolution=resolution,
                                               ref_raster=roi_clip_ref_raster)


# ################################################
# # # Process switches for GEE data merging
skip_process_data_for_CA = True
skip_process_data_for_KS = True
# ################################################

if not skip_process_data_for_CA:
    years_to_process_data_for = [2016, 2017, 2018, 2019, 2020]

    # Merging Irrigated Cropland data
    merge_gee_data_patches(data_name='Irrigated', years=years_to_process_data_for,
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/Irrigated_CA',
                           input_shape=CV_shape, resolution=CA_res,
                           gee_merge_ref_raster=gee_merge_CA_30m_refraster,
                           roi_clip_ref_raster=CA_ROI_30m_refraster, no_data=0, month_range=None)

    # Merging Rainfed Cropland data
    merge_gee_data_patches(data_name='Rainfed', years=years_to_process_data_for,
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/Rainfed_CA',
                           input_shape=CV_shape, resolution=CA_res,
                           gee_merge_ref_raster=gee_merge_CA_30m_refraster,
                           roi_clip_ref_raster=CA_ROI_30m_refraster, no_data=0, month_range=None)

    # Merging USDA CDL data
    merge_gee_data_patches(data_name='CDL', years=years_to_process_data_for,
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/CDL_CA',
                           input_shape=CV_shape, resolution=CA_res,
                           gee_merge_ref_raster=gee_merge_CA_30m_refraster,
                           roi_clip_ref_raster=CA_ROI_30m_refraster, no_data=0, month_range=None)

    # Merging OpenET Ensemble data
    merge_gee_data_patches(data_name='OpenET_ensemble', years=years_to_process_data_for,
                           month_range=(1, 12),
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/OpenET_ensemble',
                           input_shape=CV_shape, resolution=CA_res,
                           gee_merge_ref_raster=gee_merge_CA_30m_refraster,
                           roi_clip_ref_raster=CA_ROI_30m_refraster, no_data=-9999)

    # Summing OpenET for Growing season
    OpenET_monthly_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/OpenET_ensemble/merged'

    for year in years_to_process_data_for:
        datasets_to_sum = glob(os.path.join(OpenET_monthly_dir, f'*{year}_[4-9]*')) + \
                          glob(os.path.join(OpenET_monthly_dir, f'*{year}_10*'))

        print(f'summing OpenET growing season data {year}...')

        output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/OpenET_ensemble/grow_season'
        makedirs([output_dir])
        summed_raster = os.path.join(output_dir, f'OpenET_ensemble_{year}.tif')
        sum_rasters(raster_dir=None, raster_list=datasets_to_sum, output_raster=summed_raster,
                    ref_raster=CA_ROI_30m_refraster, nodata=-9999)

    # Clipping PRISM Precipitation data
    precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
    precip_datasets = glob(os.path.join(precip_dir, '*.tif'))

    print('processing PRISM precip monthly data...')
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process_data_for):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=CV_shape,
                                           output_raster_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/PRISM_precip',
                                           raster_name=None, clip_and_resample=True,
                                           resolution=CA_res, ref_raster=CA_ROI_30m_refraster)

    # Summing PRISM PPT for Before Growing season ends
    PRISM_monthly_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/PRISM_precip'

    for year in years_to_process_data_for:
        datasets_to_sum = glob(os.path.join(PRISM_monthly_dir, f'*{year}_[1-9]*')) + \
                          glob(os.path.join(PRISM_monthly_dir, f'*{year}_10*'))

        print(f'summing PRISM Precip yearly data {year}...')

        output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/PRISM_precip/yearly'
        makedirs([output_dir])
        summed_raster = os.path.join(output_dir, f'prism_precip_{year}.tif')
        sum_rasters(raster_dir=None, raster_list=datasets_to_sum, output_raster=summed_raster,
                    ref_raster=CA_ROI_30m_refraster, nodata=-9999)

    # Clipping GridMET RET data
    ret_dir = '../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly'
    ret_datasets = glob(os.path.join(ret_dir, '*.tif'))

    print('processing GridMET RET monthly data...')
    for ret in ret_datasets:
        if any(str(i) in ret for i in years_to_process_data_for):
            clip_resample_reproject_raster(input_raster=ret, input_shape=CV_shape,
                                           output_raster_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/GridMET_RET',
                                           raster_name=None, clip_and_resample=True,
                                           resolution=CA_res, ref_raster=CA_ROI_30m_refraster)

    # Summing PRISM PPT for Before Growing season ends
    RET_monthly_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/GridMET_RET'

    for year in years_to_process_data_for:
        datasets_to_sum = glob(os.path.join(RET_monthly_dir, f'*{year}_[4-9]*')) + \
                          glob(os.path.join(RET_monthly_dir, f'*{year}_10*'))

        print(f'summing GridMET RET yearly data {year}...')

        output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/GridMET_RET/yearly'
        makedirs([output_dir])
        summed_raster = os.path.join(output_dir, f'GRIDMET_RET_{year}.tif')
        sum_rasters(raster_dir=None, raster_list=datasets_to_sum, output_raster=summed_raster,
                    ref_raster=CA_ROI_30m_refraster, nodata=-9999)

if not skip_process_data_for_KS:
    years_to_process_data_for = [2016, 2017, 2018, 2019, 2020]

    # Merging Irrigated Cropland data
    merge_gee_data_patches(data_name='Irrigated', years=years_to_process_data_for,
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/Irrigated_KS',
                           input_shape=KS_shape, resolution=KS_res,
                           gee_merge_ref_raster=gee_merge_KS_30m_refraster,
                           roi_clip_ref_raster=KS_ROI_30m_refraster, no_data=0, month_range=None)

    # Merging Rainefed Cropland data
    merge_gee_data_patches(data_name='Rainfed', years=years_to_process_data_for,
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/Rainfed_KS',
                           input_shape=KS_shape, resolution=KS_res,
                           gee_merge_ref_raster=gee_merge_KS_30m_refraster,
                           roi_clip_ref_raster=KS_ROI_30m_refraster, no_data=0, month_range=None)

    # Merging USDA CDL data
    merge_gee_data_patches(data_name='CDL', years=years_to_process_data_for,
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/CDL_KS',
                           input_shape=KS_shape, resolution=KS_res,
                           gee_merge_ref_raster=gee_merge_KS_30m_refraster,
                           roi_clip_ref_raster=KS_ROI_30m_refraster, no_data=0, month_range=None)

    # Merging OpenET Ensemble data
    merge_gee_data_patches(data_name='OpenET_ensemble', years=years_to_process_data_for,
                           month_range=(1, 12),
                           input_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/OpenET_ensemble',
                           input_shape=KS_shape, resolution=KS_res,
                           gee_merge_ref_raster=gee_merge_KS_30m_refraster,
                           roi_clip_ref_raster=KS_ROI_30m_refraster, no_data=-9999)

    # Summing OpenET for Growing season
    OpenET_monthly_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/OpenET_ensemble/merged'

    for year in years_to_process_data_for:
        datasets_to_sum = glob(os.path.join(OpenET_monthly_dir, f'*{year}_[4-9]*')) + \
                          glob(os.path.join(OpenET_monthly_dir, f'*{year}_10*'))

        print(f'summing OpenET growing season data {year}...')

        output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/OpenET_ensemble/grow_season'
        makedirs([output_dir])
        summed_raster = os.path.join(output_dir, f'OpenET_ensemble_{year}.tif')
        sum_rasters(raster_dir=None, raster_list=datasets_to_sum, output_raster=summed_raster,
                    ref_raster=KS_ROI_30m_refraster, nodata=-9999)

    # Clipping PRISM Precipitation data
    precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
    precip_datasets = glob(os.path.join(precip_dir, '*.tif'))

    print('processing PRISM precip monthly data...')
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process_data_for):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=KS_shape,
                                           output_raster_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/PRISM_precip',
                                           raster_name=None, clip_and_resample=True,
                                           resolution=KS_res, ref_raster=KS_ROI_30m_refraster)

    # Summing PRISM PPT for Before Growing season ends
    PRISM_monthly_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/PRISM_precip'

    for year in years_to_process_data_for:
        datasets_to_sum = glob(os.path.join(PRISM_monthly_dir, f'*{year}_[1-9]*')) + \
                          glob(os.path.join(PRISM_monthly_dir, f'*{year}_10*'))

        print(f'summing PRISM Precip yearly data {year}...')

        output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/PRISM_precip/yearly'
        makedirs([output_dir])
        summed_raster = os.path.join(output_dir, f'prism_precip_{year}.tif')
        sum_rasters(raster_dir=None, raster_list=datasets_to_sum, output_raster=summed_raster,
                    ref_raster=KS_ROI_30m_refraster, nodata=-9999)

    # Clipping GridMET RET data
    ret_dir = '../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly'
    ret_datasets = glob(os.path.join(ret_dir, '*.tif'))

    print('processing GridMET RET monthly data...')
    for ret in ret_datasets:
        if any(str(i) in ret for i in years_to_process_data_for):
            clip_resample_reproject_raster(input_raster=ret, input_shape=KS_shape,
                                           output_raster_dir='../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/GridMET_RET',
                                           raster_name=None, clip_and_resample=True,
                                           resolution=KS_res, ref_raster=KS_ROI_30m_refraster)

    # Summing PRISM PPT for Before Growing season ends
    RET_monthly_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/GridMET_RET'

    for year in years_to_process_data_for:
        datasets_to_sum = glob(os.path.join(RET_monthly_dir, f'*{year}_[4-9]*')) + \
                          glob(os.path.join(RET_monthly_dir, f'*{year}_10*'))

        print(f'summing GridMET RET yearly data {year}...')

        output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/GridMET_RET/yearly'
        makedirs([output_dir])
        summed_raster = os.path.join(output_dir, f'GRIDMET_RET_{year}.tif')
        sum_rasters(raster_dir=None, raster_list=datasets_to_sum, output_raster=summed_raster,
                    ref_raster=KS_ROI_30m_refraster, nodata=-9999)

# ##################################################################
# # # Process switches for monthly dataframe creation for California
skip_compile_df_for_CA_monthly = True
skip_compile_df_for_CA_yearly = True
# ##################################################################

# ####### Compiling dataframe for Kansas (inside and outside GMD) at monthly scale
if not skip_compile_df_for_CA_monthly:
    # monthly data dir
    OpenET_dir_monthly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/OpenET_ensemble/merged'
    Prism_precip_dir_monthly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/PRISM_precip'
    Gridmet_dir_monthly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/GridMET_RET'

    # yearly data dir
    USDA_CDL_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/CDL_CA/merged'
    Irrigated_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/Irrigated_CA/merged'
    Rainfed_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/Rainfed_CA/merged'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    skip_df_creation = False  # #
    for year in years_to_process:
        print(f'creating monthly dataframe for CA for year {year}...')
        datasets_to_include = ['OpenET', 'PRISM_ppt', 'GridMET_RET', 'USDA_CDL', 'Irrigated', 'Rainfed']

        monthly_data_path_dict = {'OpenET': OpenET_dir_monthly,
                                  'PRISM_ppt': Prism_precip_dir_monthly,
                                  'GridMET_RET': Gridmet_dir_monthly}
        yearly_data_path_dict = {'USDA_CDL': USDA_CDL_dir,
                                 'Irrigated': Irrigated_cropland_dir,
                                 'Rainfed': Rainfed_cropland_dir}

        dataframe_path = f'../../Data_main/Data_Drivers_of_ET/CSVs/30m/CA_monthly_{year}.parquet'

        makedirs([os.path.dirname(dataframe_path)])

        create_train_test_dataframe(years_list=[year],
                                    month_range=(4, 10),
                                    monthly_data_path_dict=monthly_data_path_dict,
                                    yearly_data_path_dict=yearly_data_path_dict,
                                    static_data_path_dict=None,
                                    datasets_to_include=datasets_to_include,
                                    output_parquet=dataframe_path,
                                    skip_processing=skip_df_creation,
                                    n_partitions=5)


# ####### Compiling dataframe for Kansas (inside and outside GMD) at monthly scale
if not skip_compile_df_for_CA_yearly:
    # yearly data dir
    OpenET_dir_GrowSeason = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/OpenET_ensemble/grow_season'
    Prism_precip_dir_yearly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/PRISM_precip/yearly'
    Gridmet_dir_yearly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/GridMET_RET/yearly'

    USDA_CDL_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/CDL_CA/merged'
    Irrigated_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/Irrigated_CA/merged'
    Rainfed_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA/Rainfed_CA/merged'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    datasets_to_include = ['OpenET', 'PRISM_ppt', 'GridMET_RET', 'USDA_CDL', 'Irrigated', 'Rainfed']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/30m/CA_annual.parquet'

    makedirs([os.path.dirname(dataframe_path)])

    yearly_data_path_dict = {'OpenET': OpenET_dir_GrowSeason,
                             'PRISM_ppt': Prism_precip_dir_yearly,
                             'GridMET_RET': Gridmet_dir_yearly,
                             'USDA_CDL': USDA_CDL_dir,
                             'Irrigated': Irrigated_cropland_dir,
                             'Rainfed': Rainfed_cropland_dir}

    print(f'creating annual dataframe for CA...')
    create_train_test_dataframe(years_list=years_to_process,
                                month_range=None,
                                monthly_data_path_dict=None,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation,
                                n_partitions=5)


# ##############################################################
# # # Process switches for monthly dataframe creation for Kansas
skip_compile_df_for_KS_monthly = True
skip_compile_df_for_KS_yearly = True
# ##############################################################

# ####### Compiling dataframe for Kansas (inside and outside GMD) at monthly scale
if not skip_compile_df_for_KS_monthly:
    # monthly data dir
    OpenET_dir_monthly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/OpenET_ensemble/merged'
    Prism_precip_dir_monthly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/PRISM_precip'
    Gridmet_dir_monthly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/GridMET_RET'

    # yearly data dir
    USDA_CDL_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/CDL_KS/merged'
    Irrigated_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/Irrigated_KS/merged'
    Rainfed_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/Rainfed_KS/merged'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    skip_df_creation = False  # #
    for year in years_to_process:
        print(f'creating monthly dataframe for KS for year {year}...')
        datasets_to_include = ['OpenET', 'PRISM_ppt', 'GridMET_RET', 'USDA_CDL', 'Irrigated', 'Rainfed']

        monthly_data_path_dict = {'OpenET': OpenET_dir_monthly,
                                  'PRISM_ppt': Prism_precip_dir_monthly,
                                  'GridMET_RET': Gridmet_dir_monthly}
        yearly_data_path_dict = {'USDA_CDL': USDA_CDL_dir,
                                 'Irrigated': Irrigated_cropland_dir,
                                 'Rainfed': Rainfed_cropland_dir}

        dataframe_path = f'../../Data_main/Data_Drivers_of_ET/CSVs/30m/KS_monthly_{year}.parquet'

        makedirs([os.path.dirname(dataframe_path)])

        create_train_test_dataframe(years_list=[year],
                                    month_range=(4, 10),
                                    monthly_data_path_dict=monthly_data_path_dict,
                                    yearly_data_path_dict=yearly_data_path_dict,
                                    static_data_path_dict=None,
                                    datasets_to_include=datasets_to_include,
                                    output_parquet=dataframe_path,
                                    skip_processing=skip_df_creation,
                                    n_partitions=5)


# ####### Compiling dataframe for Kansas (inside and outside GMD) at monthly scale
if not skip_compile_df_for_KS_yearly:
    # yearly data dir
    OpenET_dir_GrowSeason = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/OpenET_ensemble/grow_season'
    Prism_precip_dir_yearly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/PRISM_precip/yearly'
    Gridmet_dir_yearly = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/GridMET_RET/yearly'

    USDA_CDL_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/CDL_KS/merged'
    Irrigated_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/Irrigated_KS/merged'
    Rainfed_cropland_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS/Rainfed_KS/merged'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    datasets_to_include = ['OpenET', 'PRISM_ppt', 'GridMET_RET', 'USDA_CDL', 'Irrigated', 'Rainfed']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/30m/KS_annual.parquet'

    makedirs([os.path.dirname(dataframe_path)])

    yearly_data_path_dict = {'OpenET': OpenET_dir_GrowSeason,
                             'PRISM_ppt': Prism_precip_dir_yearly,
                             'GridMET_RET': Gridmet_dir_yearly,
                             'USDA_CDL': USDA_CDL_dir,
                             'Irrigated': Irrigated_cropland_dir,
                             'Rainfed': Rainfed_cropland_dir}

    print(f'creating annual dataframe for KS...')
    create_train_test_dataframe(years_list=years_to_process,
                                month_range=None,
                                monthly_data_path_dict=None,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation,
                                n_partitions=5)

