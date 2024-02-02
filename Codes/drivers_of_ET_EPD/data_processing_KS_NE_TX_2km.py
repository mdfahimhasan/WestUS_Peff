import os
import sys
import numpy as np
import pandas as pd
from glob import glob

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, \
        clip_resample_reproject_raster, create_ref_rater
from Codes.utils.ml_ops import create_train_test_dataframe

# Reference resolution, shape, and rasters
no_data_value = -9999

KS_res = 0.01976293625031605786  # in deg, ~2 km
NE_res = 0.01976293625031605786  # in deg, ~2 km
TX_res = 0.01976293625031605786  # in deg, ~2 km
CA_res = 0.01976293625031605786  # in deg, ~2 km

GMD_KS_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/For_2km/ks_gmds.shp'
NE_ROI_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/For_2km/NE_ROI.shp'
TX_ROI_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/For_2km/TX_ROI.shp'
CV_CA_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/For_2km/CV.shp'

GMD_KS_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/gmd_KS_refraster.tif'
NE_ROI_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/NE_ROI_refraster.tif'
TX_ROI_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/TX_ROI_refraster.tif'
CV_CA_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/cv_CA_refraster.tif'

# Original input data directories
OpenET_monthly = '../../Data_main/Raster_data/OpenET_ensemble/WestUS_monthly'
Irrigated_cropET_monthly = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly'
Rainfed_cropET_monthly = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly'
Prism_precip_monthly = '../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
USDA_CDL_yearly = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'
Irrigated_cropland_yearly = '../../Data_main/Raster_data/Irrigated_cropland'
Rainfed_cropland_yearly = '../../Data_main/Raster_data/Rainfed_cropland'
Irrigated_cropET_grow_season = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
Rainfed_cropET_grow_season = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_grow_season'
Prism_precip_yearly = '../../Data_main/Raster_data/PRISM_Precip/WestUS'

################################################
# # Process switches for GMD, KS
skip_process_KS_data_monthly = True
################################################

if not skip_process_KS_data_monthly:
    # ####### GMD KS I
    # Clipping and resampling datasets for GMD KS
    OpenET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/OpenET_ensemble/monthly'
    Irrigated_cropET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/Irrigated_cropET/monthly'
    Rainfed_cropET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/Rainfed_cropET/monthly'
    Prism_precip_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/PRISM_Precip/monthly'
    USDA_CDL_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/USDA_CDL'
    Irrigated_cropland_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/Irrigated_cropland'
    Rainfed_cropland_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/Rainfed_cropland'
    Irrigated_cropET_grow_season_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/Irrigated_cropET/grow_season'
    Rainfed_cropET_grow_season_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/Rainfed_cropET/grow_season'
    Prism_precip_yearly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/KS/PRISM_Precip/yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_monthly_output_dir, Irrigated_cropET_monthly_output_dir, Rainfed_cropET_monthly_output_dir,
              Prism_precip_monthly_output_dir, USDA_CDL_output_dir, Irrigated_cropET_grow_season_output_dir,
              Rainfed_cropET_grow_season_output_dir])

    # monthly openET data
    openet_datasets = glob(os.path.join(OpenET_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=GMD_KS_shape,
                                       output_raster_dir=OpenET_monthly_output_dir,
                                       raster_name=None, clip_and_resample=True, resolution=KS_res,
                                       ref_raster=GMD_KS_refraster)

    # monthly irrigated cropET data
    irrig_cropET_monthly_datasets = glob(os.path.join(Irrigated_cropET_monthly, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropET_monthly_output_dir, 'interim')
    for cropET in irrig_cropET_monthly_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=GMD_KS_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=KS_res,
                                                        ref_raster=GMD_KS_refraster)

        ref_arr, ref_file = read_raster_arr_object(GMD_KS_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Irrigated_cropET_monthly_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # monthly rainfed cropET data
    rain_cropET_monthly_datasets = glob(os.path.join(Rainfed_cropET_monthly, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropET_monthly_output_dir, 'interim')
    for cropET in rain_cropET_monthly_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=GMD_KS_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=KS_res,
                                                        ref_raster=GMD_KS_refraster)

        ref_arr, ref_file = read_raster_arr_object(GMD_KS_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Rainfed_cropET_monthly_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # monthly precipitation data
    precip_datasets = glob(os.path.join(Prism_precip_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=GMD_KS_shape,
                                           output_raster_dir=Prism_precip_monthly_output_dir,
                                           raster_name=None, clip_and_resample=True, resolution=KS_res,
                                           ref_raster=GMD_KS_refraster)

    # yearly precipitation data
    precip_datasets = glob(os.path.join(Prism_precip_yearly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=GMD_KS_shape,
                                           output_raster_dir=Prism_precip_yearly_output_dir,
                                           raster_name=None, clip_and_resample=True, resolution=KS_res,
                                           ref_raster=GMD_KS_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=GMD_KS_shape,
                                           output_raster_dir=USDA_CDL_output_dir,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=KS_res, ref_raster=GMD_KS_refraster)

    irrig_cropland_datasets = glob(os.path.join(Irrigated_cropland_yearly, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropland_output_dir, 'interim')
    for cropland in irrig_cropland_datasets:
        if any(str(i) in cropland for i in years_to_process):
            clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=GMD_KS_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=KS_res, ref_raster=GMD_KS_refraster)
            ref_arr, ref_file = read_raster_arr_object(GMD_KS_refraster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            # replacing nan values where reference raster is zero with zero
            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(Irrigated_cropland_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

    rain_cropland_datasets = glob(os.path.join(Rainfed_cropland_yearly, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropland_output_dir, 'interim')
    for cropland in rain_cropland_datasets:
        if any(str(i) in cropland for i in years_to_process):
            clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=GMD_KS_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=KS_res, ref_raster=GMD_KS_refraster)

            ref_arr, ref_file = read_raster_arr_object(GMD_KS_refraster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            # replacing nan values where reference raster is zero with zero
            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(Rainfed_cropland_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

    irrig_cropET_grow_season_datasets = glob(os.path.join(Irrigated_cropET_grow_season, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropET_grow_season_output_dir, 'interim')
    for cropET in irrig_cropET_grow_season_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=GMD_KS_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=KS_res,
                                                        ref_raster=GMD_KS_refraster)

        ref_arr, ref_file = read_raster_arr_object(GMD_KS_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Irrigated_cropET_grow_season_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    rain_cropET_grow_season_datasets = glob(os.path.join(Rainfed_cropET_grow_season, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropET_grow_season_output_dir, 'interim')
    for cropET in rain_cropET_grow_season_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=GMD_KS_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=KS_res,
                                                        ref_raster=GMD_KS_refraster)

        ref_arr, ref_file = read_raster_arr_object(GMD_KS_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Rainfed_cropET_grow_season_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # Compiling monthly data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_monthly_output_dir,
                              'Irrigated_cropET': Irrigated_cropET_monthly_output_dir,
                              'Rainfed_cropET': Rainfed_cropET_monthly_output_dir,
                              'PRISM_PPT': Prism_precip_monthly_output_dir}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                             'Irrigated': Irrigated_cropland_output_dir,
                             'Rainfed': Rainfed_cropland_output_dir}

    datasets_to_include = ['OpenET', 'Irrigated_cropET', 'Rainfed_cropET', 'PRISM_PPT',
                           'USDA_CDL', 'Irrigated', 'Rainfed']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/KS_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    create_train_test_dataframe(years_list=years_to_process,
                                month_range=(4, 10),
                                monthly_data_path_dict=monthly_data_path_dict,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation)

    # Compiling yearly data to CSV
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                             'Irrigated': Irrigated_cropland_output_dir,
                             'Rainfed': Rainfed_cropland_output_dir,
                             'Irrigated_cropET': Irrigated_cropET_grow_season_output_dir,
                             'Rainfed_cropET': Rainfed_cropET_grow_season_output_dir,
                             'PRISM_PPT': Prism_precip_yearly_output_dir}

    datasets_to_include = ['Irrigated_cropET', 'Rainfed_cropET', 'USDA_CDL', 'Irrigated', 'Rainfed', 'PRISM_PPT']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/KS_yearly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    create_train_test_dataframe(years_list=years_to_process,
                                month_range=None,
                                monthly_data_path_dict=None,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation)
    # #######

################################################
# # Process switches for NE
skip_process_NE_data_monthly = True
################################################

if not skip_process_NE_data_monthly:
    # ####### NE ROI
    # Clipping and resampling datasets for NE ROI
    OpenET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/OpenET_ensemble/monthly'
    Irrigated_cropET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/Irrigated_cropET/monthly'
    Rainfed_cropET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/Rainfed_cropET/monthly'
    Prism_precip_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/PRISM_Precip/monthly'
    USDA_CDL_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/USDA_CDL'
    Irrigated_cropland_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/Irrigated_cropland'
    Rainfed_cropland_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/Rainfed_cropland'
    Irrigated_cropET_grow_season_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/Irrigated_cropET/grow_season'
    Rainfed_cropET_grow_season_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/Rainfed_cropET/grow_season'
    Prism_precip_yearly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/NE/PRISM_Precip/yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_monthly_output_dir, Irrigated_cropET_monthly_output_dir, Rainfed_cropET_monthly_output_dir,
              Prism_precip_monthly_output_dir, USDA_CDL_output_dir, Irrigated_cropET_grow_season_output_dir,
              Rainfed_cropET_grow_season_output_dir])

    # monthly openET data
    openet_datasets = glob(os.path.join(OpenET_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=NE_ROI_shape,
                                       output_raster_dir=OpenET_monthly_output_dir,
                                       raster_name=None, clip_and_resample=True, resolution=NE_res,
                                       ref_raster=NE_ROI_refraster)

    # monthly irrigated cropET data
    irrig_cropET_monthly_datasets = glob(os.path.join(Irrigated_cropET_monthly, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropET_monthly_output_dir, 'interim')
    for cropET in irrig_cropET_monthly_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=NE_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=NE_res,
                                                        ref_raster=NE_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(NE_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Irrigated_cropET_monthly_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # monthly rainfed cropET data
    rain_cropET_monthly_datasets = glob(os.path.join(Rainfed_cropET_monthly, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropET_monthly_output_dir, 'interim')
    for cropET in rain_cropET_monthly_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=NE_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=NE_res,
                                                        ref_raster=NE_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(NE_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Rainfed_cropET_monthly_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # monthly precipitation data
    precip_datasets = glob(os.path.join(Prism_precip_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=NE_ROI_shape,
                                           output_raster_dir=Prism_precip_monthly_output_dir,
                                           raster_name=None, clip_and_resample=True, resolution=NE_res,
                                           ref_raster=NE_ROI_refraster)

    # yearly precipitation data
    precip_datasets = glob(os.path.join(Prism_precip_yearly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=NE_ROI_shape,
                                           output_raster_dir=Prism_precip_yearly_output_dir,
                                           raster_name=None, clip_and_resample=True, resolution=NE_res,
                                           ref_raster=NE_ROI_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=NE_ROI_shape,
                                           output_raster_dir=USDA_CDL_output_dir,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=NE_res, ref_raster=NE_ROI_refraster)

    irrig_cropland_datasets = glob(os.path.join(Irrigated_cropland_yearly, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropland_output_dir, 'interim')
    for cropland in irrig_cropland_datasets:
        if any(str(i) in cropland for i in years_to_process):
            clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=NE_ROI_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=NE_res, ref_raster=NE_ROI_refraster)
            ref_arr, ref_file = read_raster_arr_object(NE_ROI_refraster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            # replacing nan values where reference raster is zero with zero
            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(Irrigated_cropland_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

    rain_cropland_datasets = glob(os.path.join(Rainfed_cropland_yearly, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropland_output_dir, 'interim')
    for cropland in rain_cropland_datasets:
        if any(str(i) in cropland for i in years_to_process):
            clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=NE_ROI_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=NE_res, ref_raster=NE_ROI_refraster)

            ref_arr, ref_file = read_raster_arr_object(NE_ROI_refraster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            # replacing nan values where reference raster is zero with zero
            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(Rainfed_cropland_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

    irrig_cropET_grow_season_datasets = glob(os.path.join(Irrigated_cropET_grow_season, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropET_grow_season_output_dir, 'interim')
    for cropET in irrig_cropET_grow_season_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=NE_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=NE_res,
                                                        ref_raster=NE_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(NE_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Irrigated_cropET_grow_season_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    rain_cropET_grow_season_datasets = glob(os.path.join(Rainfed_cropET_grow_season, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropET_grow_season_output_dir, 'interim')
    for cropET in rain_cropET_grow_season_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=NE_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=NE_res,
                                                        ref_raster=NE_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(NE_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Rainfed_cropET_grow_season_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # Compiling monthly data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_monthly_output_dir,
                              'Irrigated_cropET': Irrigated_cropET_monthly_output_dir,
                              'Rainfed_cropET': Rainfed_cropET_monthly_output_dir,
                              'PRISM_PPT': Prism_precip_monthly_output_dir}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                             'Irrigated': Irrigated_cropland_output_dir,
                             'Rainfed': Rainfed_cropland_output_dir}

    datasets_to_include = ['OpenET', 'Irrigated_cropET', 'Rainfed_cropET', 'PRISM_PPT',
                           'USDA_CDL', 'Irrigated', 'Rainfed']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/NE_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    create_train_test_dataframe(years_list=years_to_process,
                                month_range=(4, 10),
                                monthly_data_path_dict=monthly_data_path_dict,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation)

    # Compiling yearly data to CSV
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                             'Irrigated': Irrigated_cropland_output_dir,
                             'Rainfed': Rainfed_cropland_output_dir,
                             'Irrigated_cropET': Irrigated_cropET_grow_season_output_dir,
                             'Rainfed_cropET': Rainfed_cropET_grow_season_output_dir,
                             'PRISM_PPT': Prism_precip_yearly_output_dir}

    datasets_to_include = ['Irrigated_cropET', 'Rainfed_cropET', 'USDA_CDL', 'Irrigated', 'Rainfed', 'PRISM_PPT']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/NE_yearly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    create_train_test_dataframe(years_list=years_to_process,
                                month_range=None,
                                monthly_data_path_dict=None,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation)
    # #######


################################################
# # Process switches for TX
skip_process_TX_data_monthly = True
################################################

if not skip_process_TX_data_monthly:
    # ####### TX ROI
    # Clipping and resampling datasets for TX ROI
    OpenET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/OpenET_ensemble/monthly'
    Irrigated_cropET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/Irrigated_cropET/monthly'
    Rainfed_cropET_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/Rainfed_cropET/monthly'
    Prism_precip_monthly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/PRISM_Precip/monthly'
    USDA_CDL_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/USDA_CDL'
    Irrigated_cropland_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/Irrigated_cropland'
    Rainfed_cropland_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/Rainfed_cropland'
    Irrigated_cropET_grow_season_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/Irrigated_cropET/grow_season'
    Rainfed_cropET_grow_season_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/Rainfed_cropET/grow_season'
    Prism_precip_yearly_output_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/2km/TX/PRISM_Precip/yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_monthly_output_dir, Irrigated_cropET_monthly_output_dir, Rainfed_cropET_monthly_output_dir,
              Prism_precip_monthly_output_dir, USDA_CDL_output_dir, Irrigated_cropET_grow_season_output_dir,
              Rainfed_cropET_grow_season_output_dir])

    # monthly openET data
    openet_datasets = glob(os.path.join(OpenET_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=TX_ROI_shape,
                                       output_raster_dir=OpenET_monthly_output_dir,
                                       raster_name=None, clip_and_resample=True, resolution=TX_res,
                                       ref_raster=TX_ROI_refraster)

    # monthly irrigated cropET data
    irrig_cropET_monthly_datasets = glob(os.path.join(Irrigated_cropET_monthly, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropET_monthly_output_dir, 'interim')
    for cropET in irrig_cropET_monthly_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=TX_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=TX_res,
                                                        ref_raster=TX_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(TX_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Irrigated_cropET_monthly_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # monthly rainfed cropET data
    rain_cropET_monthly_datasets = glob(os.path.join(Rainfed_cropET_monthly, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropET_monthly_output_dir, 'interim')
    for cropET in rain_cropET_monthly_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=TX_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=TX_res,
                                                        ref_raster=TX_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(TX_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Rainfed_cropET_monthly_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # monthly precipitation data
    precip_datasets = glob(os.path.join(Prism_precip_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=TX_ROI_shape,
                                           output_raster_dir=Prism_precip_monthly_output_dir,
                                           raster_name=None, clip_and_resample=True, resolution=TX_res,
                                           ref_raster=TX_ROI_refraster)

    # yearly precipitation data
    precip_datasets = glob(os.path.join(Prism_precip_yearly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=TX_ROI_shape,
                                           output_raster_dir=Prism_precip_yearly_output_dir,
                                           raster_name=None, clip_and_resample=True, resolution=TX_res,
                                           ref_raster=TX_ROI_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=TX_ROI_shape,
                                           output_raster_dir=USDA_CDL_output_dir,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=TX_res, ref_raster=TX_ROI_refraster)

    irrig_cropland_datasets = glob(os.path.join(Irrigated_cropland_yearly, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropland_output_dir, 'interim')
    for cropland in irrig_cropland_datasets:
        if any(str(i) in cropland for i in years_to_process):
            clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=TX_ROI_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=TX_res, ref_raster=TX_ROI_refraster)
            ref_arr, ref_file = read_raster_arr_object(TX_ROI_refraster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            # replacing nan values where reference raster is zero with zero
            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(Irrigated_cropland_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

    rain_cropland_datasets = glob(os.path.join(Rainfed_cropland_yearly, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropland_output_dir, 'interim')
    for cropland in rain_cropland_datasets:
        if any(str(i) in cropland for i in years_to_process):
            clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=TX_ROI_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=TX_res, ref_raster=TX_ROI_refraster)

            ref_arr, ref_file = read_raster_arr_object(TX_ROI_refraster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            # replacing nan values where reference raster is zero with zero
            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(Rainfed_cropland_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

    irrig_cropET_grow_season_datasets = glob(os.path.join(Irrigated_cropET_grow_season, '*.tif'))
    clipped_output_dir = os.path.join(Irrigated_cropET_grow_season_output_dir, 'interim')
    for cropET in irrig_cropET_grow_season_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=TX_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=TX_res,
                                                        ref_raster=TX_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(TX_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Irrigated_cropET_grow_season_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    rain_cropET_grow_season_datasets = glob(os.path.join(Rainfed_cropET_grow_season, '*.tif'))
    clipped_output_dir = os.path.join(Rainfed_cropET_grow_season_output_dir, 'interim')
    for cropET in rain_cropET_grow_season_datasets:
        clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=TX_ROI_shape,
                                                        output_raster_dir=clipped_output_dir,
                                                        raster_name=None, clip_and_resample=True, resolution=TX_res,
                                                        ref_raster=TX_ROI_refraster)

        ref_arr, ref_file = read_raster_arr_object(TX_ROI_refraster)
        clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

        # replacing nan values where reference raster is zero with zero
        clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

        output_raster = os.path.join(Rainfed_cropET_grow_season_output_dir, os.path.basename(clipped_raster))
        write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                              output_path=output_raster)

    # Compiling monthly data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_monthly_output_dir,
                              'Irrigated_cropET': Irrigated_cropET_monthly_output_dir,
                              'Rainfed_cropET': Rainfed_cropET_monthly_output_dir,
                              'PRISM_PPT': Prism_precip_monthly_output_dir}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                             'Irrigated': Irrigated_cropland_output_dir,
                             'Rainfed': Rainfed_cropland_output_dir}

    datasets_to_include = ['OpenET', 'Irrigated_cropET', 'Rainfed_cropET', 'PRISM_PPT',
                           'USDA_CDL', 'Irrigated', 'Rainfed']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/TX_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    create_train_test_dataframe(years_list=years_to_process,
                                month_range=(4, 10),
                                monthly_data_path_dict=monthly_data_path_dict,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation)

    # Compiling yearly data to CSV
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                             'Irrigated': Irrigated_cropland_output_dir,
                             'Rainfed': Rainfed_cropland_output_dir,
                             'Irrigated_cropET': Irrigated_cropET_grow_season_output_dir,
                             'Rainfed_cropET': Rainfed_cropET_grow_season_output_dir,
                             'PRISM_PPT': Prism_precip_yearly_output_dir}

    datasets_to_include = ['Irrigated_cropET', 'Rainfed_cropET', 'USDA_CDL', 'Irrigated', 'Rainfed', 'PRISM_PPT']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/TX_yearly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    create_train_test_dataframe(years_list=years_to_process,
                                month_range=None,
                                monthly_data_path_dict=None,
                                yearly_data_path_dict=yearly_data_path_dict,
                                static_data_path_dict=None,
                                datasets_to_include=datasets_to_include,
                                output_parquet=dataframe_path,
                                skip_processing=skip_df_creation)
    # #######

# ##############################################################
# # # Process switches for adding data to KS GMD4 annual pumping csv
skip_add_data_to_KS_annual = True
# ##############################################################

if not skip_add_data_to_KS_annual:
    years = [2016, 2017, 2018, 2019, 2020]
    irrigated_cropET_grow_season = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
    openET_total_grow_season = '../../Data_main/Raster_data/OpenET_ensemble/WestUS_grow_season'

    pumping_df = pd.read_csv('../../Data_main/Data_Drivers_of_ET/CSVs/KS_GW_GMD4.csv')

    pumping_df_new = pd.DataFrame()  # an empty dataframe to add data by year (with newly extracted irrigated cropET data)

    for year in years:
        df = pumping_df[pumping_df['Year'] == year].copy()   # dataframe selected for a particular yar
        irr_cropET_val = []   # empty list to store extracted irrigated cropET data
        totalET_val = []  # empty list to store extracted total ET ata
        for index, row in df.iterrows():
            # extracting irrigated cropET data
            irr_cropET_data = glob(os.path.join(irrigated_cropET_grow_season, f"*{int(row['Year'])}*.tif"))[0]

            irr_cropET_arr, irr_cropET_file = read_raster_arr_object(irr_cropET_data)
            x, y = irr_cropET_file.index(row['long_nad83'], row['lat_nad83'])
            irr_et_value = irr_cropET_arr[x, y]
            irr_cropET_val.append(irr_et_value)

            # extracting total ET data
            totalET_data = glob(os.path.join(openET_total_grow_season, f"*{int(row['Year'])}*.tif"))[0]

            totalET_arr, totalET_file = read_raster_arr_object(totalET_data)
            x, y = totalET_file.index(row['long_nad83'], row['lat_nad83'])
            total_et_value = totalET_arr[x, y]
            totalET_val.append(total_et_value)

        # Appending extracted irrigated cropET and total ET data to the dataframe
        df.loc[:, 'Irrigated_cropET'] = irr_cropET_val
        df.loc[:, 'OpenET'] = totalET_val
        df = df.dropna()
        pumping_df_new = pd.concat([pumping_df_new, df], axis=0)

    # saving the updated dataframe
    pumping_df_new.to_csv('../../Data_main/Data_Drivers_of_ET/CSVs/KS_GW_GMD4_updated.csv', index=False)
