import os
import sys
from glob import glob

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import clip_resample_reproject_raster
from Codes.utils.ml_ops import create_train_test_dataframe

# Reference resolution, shape, and rasters
no_data_value = -9999
GMD_KS_irrig_res = 0.01976293625104601215  # in deg, ~2 km
GMD_KS_rain_res = 0.01976293625031605786  # in deg, ~2 km
CV_CA_irrig_res = 0.0197629362503160578  # in deg, ~2 km
CV_CA_rain_res = 0.0197629362500000294  # in deg, ~2 km

GMD_KS_irrig_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/ks_gmds.shp'
GMD_KS_rain_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/KS_rain.shp'
CV_CA_irrig_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/CV_irrig.shp'
CV_CA_rain_shape = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/ROIs/CV_rain.shp'

GMD_KS_irrig_refraster = '../../Data_main/refrasters/2km/GMD_KS_irrig_refraster.tif'
GMD_KS_rain_refraster = '../../Data_main/refrasters/2km/KS_rain_refraster.tif'
CV_CA_irrig_refraster = '../../Data_main/refrasters/2km/CV_CA_irrig_refraster.tif'
CV_CA_rain_refraster = '../../Data_main/refrasters/2km/CV_CA_rain_refraster.tif'


OpenET_input_dir_monthly = '../../Data_main/Raster_data/OpenET_ensemble/WestUS_monthly'
Prism_precip_input_dir_monthly = '../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
USDA_CDL_input_dir_yearly = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'

################################################
# # Process switches for monthly dataframe
skip_process_GMD_KS_irrig_data_monthly = True
skip_process_KS_rain_data_monthly = True
skip_process_CV_CA_irrig_data_monthly = True
skip_process_CV_CA_rain_data_monthly = True
################################################


if not skip_process_GMD_KS_irrig_data_monthly:
    # ####### GMD KS Irrigated Region
    # Clipping and resampling datasets for GMD KS irrigated region
    OpenET_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/OpenET_ensemble/WestUS_monthly'
    Prism_precip_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/PRISM_Precip/WestUS_monthly'
    USDA_CDL_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/USDA_CDL/WestUS_yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_output_dir_KS_irrig, Prism_precip_output_dir_KS_irrig, USDA_CDL_output_dir_KS_irrig])

    openet_datasets = glob(os.path.join(OpenET_input_dir_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=GMD_KS_irrig_shape,
                                       output_raster_dir=OpenET_output_dir_KS_irrig,
                                       raster_name=None, clip_and_resample=True, resolution=GMD_KS_irrig_res,
                                       ref_raster=GMD_KS_irrig_refraster)

    precip_datasets = glob(os.path.join(Prism_precip_input_dir_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=GMD_KS_irrig_shape,
                                           output_raster_dir=Prism_precip_output_dir_KS_irrig,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=GMD_KS_irrig_res, ref_raster=GMD_KS_irrig_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_input_dir_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=GMD_KS_irrig_shape,
                                           output_raster_dir=USDA_CDL_output_dir_KS_irrig,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=GMD_KS_irrig_res, ref_raster=GMD_KS_irrig_refraster)

    # Compiling data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_output_dir_KS_irrig,
                              'PRISM_PPT': Prism_precip_output_dir_KS_irrig}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir_KS_irrig}

    datasets_to_include = ['OpenET', 'PRISM_PPT', 'USDA_CDL']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/GMD_KS_irrig_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    compiled_parquet = create_train_test_dataframe(years_list=years_to_process,
                                                   month_range=(4, 10),
                                                   monthly_data_path_dict=monthly_data_path_dict,
                                                   yearly_data_path_dict=yearly_data_path_dict,
                                                   static_data_path_dict=None,
                                                   datasets_to_include=datasets_to_include,
                                                   output_parquet=dataframe_path,
                                                   skip_processing=skip_df_creation)

    # #######

if not skip_process_KS_rain_data_monthly:
    # ####### KS Rainfed Region
    # Clipping and resampling datasets for KS Rainfed region
    OpenET_output_dir_KS_rain = '../../Data_main/Data_Drivers_of_ET/Raster_data/KS_rain/OpenET_ensemble/WestUS_monthly'
    Prism_precip_output_dir_KS_rain = '../../Data_main/Data_Drivers_of_ET/Raster_data/KS_rain/PRISM_Precip/WestUS_monthly'
    USDA_CDL_output_dir_KS_rain = '../../Data_main/Data_Drivers_of_ET/Raster_data/KS_rain/USDA_CDL/WestUS_yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_output_dir_KS_rain, Prism_precip_output_dir_KS_rain, USDA_CDL_output_dir_KS_rain])

    openet_datasets = glob(os.path.join(OpenET_input_dir_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=GMD_KS_rain_shape,
                                       output_raster_dir=OpenET_output_dir_KS_rain,
                                       raster_name=None, clip_and_resample=True, resolution=GMD_KS_rain_res,
                                       ref_raster=GMD_KS_rain_refraster)

    precip_datasets = glob(os.path.join(Prism_precip_input_dir_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=GMD_KS_rain_shape,
                                           output_raster_dir=Prism_precip_output_dir_KS_rain,
                                           raster_name=None, clip_and_resample=True, resolution=GMD_KS_rain_res,
                                           ref_raster=GMD_KS_rain_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_input_dir_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=GMD_KS_rain_shape,
                                           output_raster_dir=USDA_CDL_output_dir_KS_rain,
                                           raster_name=None, clip_and_resample=True, resolution=GMD_KS_rain_res,
                                           ref_raster=GMD_KS_rain_refraster)

    # Compiling data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_output_dir_KS_rain,
                              'PRISM_PPT': Prism_precip_output_dir_KS_rain}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir_KS_rain}

    datasets_to_include = ['OpenET', 'PRISM_PPT', 'USDA_CDL']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/KS_rain_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    compiled_parquet = create_train_test_dataframe(years_list=years_to_process,
                                                   month_range=(4, 10),
                                                   monthly_data_path_dict=monthly_data_path_dict,
                                                   yearly_data_path_dict=yearly_data_path_dict,
                                                   static_data_path_dict=None,
                                                   datasets_to_include=datasets_to_include,
                                                   output_parquet=dataframe_path,
                                                   skip_processing=skip_df_creation)
    # #######

if not skip_process_CV_CA_irrig_data_monthly:
    # ####### CV Irrigated Region
    # Clipping and resampling datasets for CV irrigated region
    OpenET_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/OpenET_ensemble/WestUS_monthly'
    Prism_precip_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/PRISM_Precip/WestUS_monthly'
    USDA_CDL_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/USDA_CDL/WestUS_yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_output_dir_CA_irrig, Prism_precip_output_dir_CA_irrig, USDA_CDL_output_dir_CA_irrig])

    openet_datasets = glob(os.path.join(OpenET_input_dir_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=CV_CA_irrig_shape,
                                       output_raster_dir=OpenET_output_dir_CA_irrig,
                                       raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                       ref_raster=CV_CA_irrig_refraster)

    precip_datasets = glob(os.path.join(Prism_precip_input_dir_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=CV_CA_irrig_shape,
                                           output_raster_dir=Prism_precip_output_dir_CA_irrig,
                                           raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                           ref_raster=CV_CA_irrig_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_input_dir_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=CV_CA_irrig_shape,
                                           output_raster_dir=USDA_CDL_output_dir_CA_irrig,
                                           raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                           ref_raster=CV_CA_irrig_refraster)

    # Compiling data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_output_dir_CA_irrig,
                              'PRISM_PPT': Prism_precip_output_dir_CA_irrig}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir_CA_irrig}

    datasets_to_include = ['OpenET', 'PRISM_PPT', 'USDA_CDL']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/CV_CA_irrig_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    compiled_parquet = create_train_test_dataframe(years_list=years_to_process,
                                                   month_range=(4, 10),
                                                   monthly_data_path_dict=monthly_data_path_dict,
                                                   yearly_data_path_dict=yearly_data_path_dict,
                                                   static_data_path_dict=None,
                                                   datasets_to_include=datasets_to_include,
                                                   output_parquet=dataframe_path,
                                                   skip_processing=skip_df_creation)
    # #######

if not skip_process_CV_CA_rain_data_monthly:
    # ####### CV Rainfed Region
    # Clipping and resampling datasets for CV irrigated region
    OpenET_output_dir_CA_rain = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_rain/OpenET_ensemble/WestUS_monthly'
    Prism_precip_output_dir_CA_rain = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_rain/PRISM_Precip/WestUS_monthly'
    USDA_CDL_output_dir_CA_rain = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_rain/USDA_CDL/WestUS_yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_output_dir_CA_rain, Prism_precip_output_dir_CA_rain, USDA_CDL_output_dir_CA_rain])

    openet_datasets = glob(os.path.join(OpenET_input_dir_monthly, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=CV_CA_rain_shape,
                                       output_raster_dir=OpenET_output_dir_CA_rain,
                                       raster_name=None, clip_and_resample=True, resolution=CV_CA_rain_res,
                                       ref_raster=CV_CA_rain_refraster)

    precip_datasets = glob(os.path.join(Prism_precip_input_dir_monthly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=CV_CA_rain_shape,
                                           output_raster_dir=Prism_precip_output_dir_CA_rain,
                                           raster_name=None, clip_and_resample=True, resolution=CV_CA_rain_res,
                                           ref_raster=CV_CA_rain_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_input_dir_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=CV_CA_rain_shape,
                                           output_raster_dir=USDA_CDL_output_dir_CA_rain,
                                           raster_name=None, clip_and_resample=True, resolution=CV_CA_rain_res,
                                           ref_raster=CV_CA_rain_refraster)

    # Compiling data to CSV
    monthly_data_path_dict = {'OpenET': OpenET_output_dir_CA_rain,
                              'PRISM_PPT': Prism_precip_output_dir_CA_rain}
    yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir_CA_rain}

    datasets_to_include = ['OpenET', 'PRISM_PPT', 'USDA_CDL']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/CV_CA_rain_monthly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    compiled_parquet = create_train_test_dataframe(years_list=years_to_process,
                                                   month_range=(4, 10),
                                                   monthly_data_path_dict=monthly_data_path_dict,
                                                   yearly_data_path_dict=yearly_data_path_dict,
                                                   static_data_path_dict=None,
                                                   datasets_to_include=datasets_to_include,
                                                   output_parquet=dataframe_path,
                                                   skip_processing=skip_df_creation)
    # #######

# **********************************************************************************************************************
# **********************************************************************************************************************
OpenET_input_dir_grow_season = '../../Data_main/Raster_data/OpenET_ensemble/WestUS_yearly'
Prism_precip_input_dir_yearly = '../../Data_main/Raster_data/PRISM_Precip/WestUS'
GridMET_RET_input_dir_grow_season = '../../Data_main/Raster_data/GRIDMET_RET/WestUS_yearly'
USDA_CDL_input_dir_yearly = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'

################################################
# # Process switches for yearly dataframe
skip_process_GMD_KS_irrig_data_yearly = True
skip_process_CV_CA_irrig_data_yearly = True
################################################


if not skip_process_GMD_KS_irrig_data_yearly:
    # ####### GMD KS Irrigated Region
    # Clipping and resampling datasets for GMD KS irrigated region
    OpenET_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/OpenET_ensemble/WestUS_grow_season'
    Prism_precip_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/PRISM_Precip/WestUS_yearly'
    GRIDMET_RET_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/GRIDMET_RET/WestUS_grow_season'
    USDA_CDL_output_dir_KS_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/GMD_KS_irrig/USDA_CDL/WestUS_yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_output_dir_KS_irrig, Prism_precip_output_dir_KS_irrig, USDA_CDL_output_dir_KS_irrig])

    openet_datasets = glob(os.path.join(OpenET_input_dir_grow_season, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=GMD_KS_irrig_shape,
                                       output_raster_dir=OpenET_output_dir_KS_irrig,
                                       raster_name=None, clip_and_resample=True, resolution=GMD_KS_irrig_res,
                                       ref_raster=GMD_KS_irrig_refraster)

    ret_datasets = glob(os.path.join(GridMET_RET_input_dir_grow_season, '*.tif'))
    for ret in ret_datasets:
        clip_resample_reproject_raster(input_raster=ret, input_shape=GMD_KS_irrig_shape,
                                       output_raster_dir=GRIDMET_RET_output_dir_KS_irrig,
                                       raster_name=None, clip_and_resample=True, resolution=GMD_KS_irrig_res,
                                       ref_raster=GMD_KS_irrig_refraster)

    precip_datasets = glob(os.path.join(Prism_precip_input_dir_yearly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=GMD_KS_irrig_shape,
                                           output_raster_dir=Prism_precip_output_dir_KS_irrig,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=GMD_KS_irrig_res, ref_raster=GMD_KS_irrig_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_input_dir_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=GMD_KS_irrig_shape,
                                           output_raster_dir=USDA_CDL_output_dir_KS_irrig,
                                           raster_name=None, clip_and_resample=True,
                                           resolution=GMD_KS_irrig_res, ref_raster=GMD_KS_irrig_refraster)

    # Compiling data to CSV
    data_path_dict = {'OpenET': OpenET_output_dir_KS_irrig,
                      'PRISM_PPT': Prism_precip_output_dir_KS_irrig,
                      'GRIDMET_RET': GRIDMET_RET_output_dir_KS_irrig,
                      'USDA_CDL': USDA_CDL_output_dir_KS_irrig}

    datasets_to_include = ['OpenET', 'PRISM_PPT', 'GRIDMET_RET', 'USDA_CDL']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/GMD_KS_irrig_yearly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    compiled_parquet = create_train_test_dataframe(years_list=years_to_process,
                                                           month_range=None, monthly_data_path_dict=None,
                                                           yearly_data_path_dict=data_path_dict,
                                                           static_data_path_dict=None,
                                                           datasets_to_include=datasets_to_include,
                                                           output_parquet=dataframe_path,
                                                           skip_processing=skip_df_creation)

    #######

if not skip_process_CV_CA_irrig_data_yearly:
    # ####### CV Irrigated Region
    # Clipping and resampling datasets for CV irrigated region
    OpenET_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/OpenET_ensemble/WestUS_grow_season'
    Prism_precip_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/PRISM_Precip/WestUS_yearly'
    GRIDMET_RET_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/GRIDMET_RET/WestUS_grow_season'
    USDA_CDL_output_dir_CA_irrig = '../../Data_main/Data_Drivers_of_ET/Raster_data/CV_CA_irrig/USDA_CDL/WestUS_yearly'

    years_to_process = [2016, 2017, 2018, 2019, 2020]

    makedirs([OpenET_output_dir_CA_irrig, Prism_precip_output_dir_CA_irrig, USDA_CDL_output_dir_CA_irrig])

    openet_datasets = glob(os.path.join(OpenET_input_dir_grow_season, '*.tif'))
    for et in openet_datasets:
        clip_resample_reproject_raster(input_raster=et, input_shape=CV_CA_irrig_shape,
                                       output_raster_dir=OpenET_output_dir_CA_irrig,
                                       raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                       ref_raster=CV_CA_irrig_refraster)

    precip_datasets = glob(os.path.join(Prism_precip_input_dir_yearly, '*.tif'))
    for ppt in precip_datasets:
        if any(str(i) in ppt for i in years_to_process):
            clip_resample_reproject_raster(input_raster=ppt, input_shape=CV_CA_irrig_shape,
                                           output_raster_dir=Prism_precip_output_dir_CA_irrig,
                                           raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                           ref_raster=CV_CA_irrig_refraster)

    ret_datasets = glob(os.path.join(GridMET_RET_input_dir_grow_season, '*.tif'))
    for ret in ret_datasets:
        clip_resample_reproject_raster(input_raster=ret, input_shape=CV_CA_irrig_shape,
                                       output_raster_dir=GRIDMET_RET_output_dir_CA_irrig,
                                       raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                       ref_raster=CV_CA_irrig_refraster)

    cdl_datasets = glob(os.path.join(USDA_CDL_input_dir_yearly, '*.tif'))
    for cdl in cdl_datasets:
        if any(str(i) in cdl for i in years_to_process):
            clip_resample_reproject_raster(input_raster=cdl, input_shape=CV_CA_irrig_shape,
                                           output_raster_dir=USDA_CDL_output_dir_CA_irrig,
                                           raster_name=None, clip_and_resample=True, resolution=CV_CA_irrig_res,
                                           ref_raster=CV_CA_irrig_refraster)

    # Compiling data to CSV
    data_path_dict = {'OpenET': OpenET_output_dir_CA_irrig,
                      'PRISM_PPT': Prism_precip_output_dir_CA_irrig,
                      'GRIDMET_RET': GRIDMET_RET_output_dir_CA_irrig,
                      'USDA_CDL': USDA_CDL_output_dir_CA_irrig}

    datasets_to_include = ['OpenET', 'PRISM_PPT', 'GRIDMET_RET', 'USDA_CDL']

    skip_df_creation = False  # #
    dataframe_path = '../../Data_main/Data_Drivers_of_ET/CSVs/2km/CV_CA_irrig_yearly.csv'

    makedirs([os.path.dirname(dataframe_path)])

    compiled_parquet = create_train_test_dataframe(years_list=years_to_process,
                                                           month_range=None, monthly_data_path_dict=None,
                                                           yearly_data_path_dict=data_path_dict,
                                                           static_data_path_dict=None,
                                                           datasets_to_include=datasets_to_include,
                                                           output_parquet=dataframe_path,
                                                           skip_processing=skip_df_creation)
    # #######

# import numpy as np
# from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster
# arr, file = read_raster_arr_object('F:/WestUS_NetGW/scratch/gee_merge_refraster_CA_30m.tif')
#
# arr = np.where(np.isnan(arr), -9999, 0)
# write_array_to_raster(arr, file, file.transform, 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/refrasters/30m/gee_merge_refraster_CA_30m.tif')