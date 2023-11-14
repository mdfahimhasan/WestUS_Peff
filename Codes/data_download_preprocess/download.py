import os
import ee
import sys
import zipfile
import requests
import itertools
from glob import glob
import geopandas as gpd
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, mosaic_rasters_from_directory, clip_resample_reproject_raster

# ee.Authenticate()

# # if ee.Authenticate() doesn't work or gives 'gcloud' error_mae go to the
# 'https://developers.google.com/earth-engine/guides/python_install' link and follow instructions for
# local/remote machine.
# In local machine, install gcloud (if not already installed) and follow the steps from the link to authenticate.
# For remote machine, https://www.youtube.com/watch?v=k-8qFh8EfFA this link was helpful for installing gcloud.
# Couldn't figure out how to authenticate in server pc, will try with new workstation
# If authenticated, no need to run the authentication process again. Just start from ee.initialize()

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def download_from_url(download_dir, url_list):
    """
    Download Data from url.

    :param download_dir: File path to save downloaded data.
    :param url_list: A list of url/urls to download.

    :return:
    """
    makedirs([download_dir])
    for url in url_list:
        fname = url.rsplit('/', 1)[1]
        print(f'Downloading {fname}......')
        r = requests.get(url, allow_redirects=True)
        download_fname = os.path.join(download_dir, fname)
        open(download_fname, 'wb').write(r.content)
        print('Download complete')


def extract_data(zip_dir_or_list, out_dir, search_by='*.zip', rename_file=True):
    """
    Extract zipped file.

    :param zip_dir_or_list: (folder path or list). WIll automatically detect directory path ot list.
                            Directory path of zipped file.
                            Alternatively, a List of file paths to extract.
    :param out_dir: File path where data will be extracted.
    :param search_by: Keyword for searching files, default is '*.zip'.
    :param rename_file: True if file rename is required while extracting.

    :return: List of zipped files (can be used for deleting all the files). This function will unzip these files.
    """
    print('Extracting zip files.....')

    makedirs([out_dir])
    if type(zip_dir_or_list) is list:  # If input is a list of zipped file paths
        all_zipped_files = zip_dir_or_list
    else:  # if input is a file path of directory
        all_zipped_files = glob(os.path.join(zip_dir_or_list, search_by))

    for zip_file in all_zipped_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_file:
                zip_key = zip_file[zip_file.rfind(os.sep) + 1:zip_file.rfind('.')]
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_key + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)
    return all_zipped_files


def get_gee_dict(data_name):
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    gee_data_dict = {
        'SMAP_SM': 'NASA_USDA/HSL/SMAP10KM_soil_moisture',
        'LANDSAT_NDWI': 'LANDSAT/LC08/C01/T1_8DAY_NDWI',  # check for cloudcover
        'LANDSAT_NDVI': 'LANDSAT/LC08/C01/T1_8DAY_NDVI',  # check for cloudcover
        'GPM_Precip': 'NASA/GPM_L3/IMERG_MONTHLY_V06',
        'GRIDMET_Precip': 'IDAHO_EPSCOR/GRIDMET',
        'MODIS_Day_LST': 'MODIS/006/MOD11A2',  # check for cloudcover
        'MODIS_Terra_NDVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_Terra_EVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_NDWI': 'MODIS/006/MOD09A1',  # cloudcover mask added later
        'MODIS_NDVI': 'MODIS/006/MOD09A1',  # cloudcover mask added later
        'MODIS_LAI': 'MODIS/006/MCD15A3H',
        'MODIS_ET': 'MODIS/006/MOD16A2',  # unit in kg/m2
        'TERRACLIMATE_ET': 'IDAHO_EPSCOR/TERRACLIMATE',
        'OpenET_ensemble': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'Irrig_crop_OpenET_IrrMapper': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'Irrig_crop_OpenET_LANID': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'Rainfed_crop_OpenET_IrrMapper': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'Rainfed_crop_OpenET_LANID': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'TERRACLIMATE_RET': 'IDAHO_EPSCOR/TERRACLIMATE',
        'TERRACLIMATE_vap_pres': 'IDAHO_EPSCOR/TERRACLIMATE',
        'TERRACLIMATE_vap_pres_def': 'IDAHO_EPSCOR/TERRACLIMATE',
        'GRIDMET_RET': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_max_RH': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_min_RH': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_wind_vel': 'IDAHO_EPSCOR/GRIDMET',  # at 10m
        'GRIDMET_short_rad': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_vap_pres_def': 'IDAHO_EPSCOR/GRIDMET',
        'DAYMET_sun_hr': 'NASA/ORNL/DAYMET_V4',
        'USDA_CDL': 'USDA/NASS/CDL',
        'IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
        'LANID': 'projects/ee-fahim/assets/LANID_for_selected_states/selected_Annual_LANID',
        'AIM-HPA': 'projects/ee-fahim/assets/AIM-HPA/AIM_HPA_Annual',
        'Irrigation_Frac_IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
        'Irrigation_Frac_LANID': 'projects/ee-fahim/assets/LANID_for_selected_states/selected_Annual_LANID',
        'Rainfed_Frac_IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
        'Rainfed_Frac_LANID': 'projects/ee-fahim/assets/LANID_for_selected_states/selected_Annual_LANID',
        'Field_capacity': 'OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01',
        'Bulk_density': 'OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02',
        'Organic_carbon_content': 'OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02',
        'Sand_content': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
        'Clay_content': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'DEM': 'USGS/SRTMGL1_003',
        'Tree_cover': 'NASA/MEASURES/GFCC/TC/v3'
    }

    gee_band_dict = {
        'SMAP_SM': 'ssm',
        'LANDSAT_NDWI': 'NDWI',
        'LANDSAT_NDVI': 'NDVI',
        'GPM_Precip': 'precipitation',
        'GRIDMET_Precip': 'pr',
        'MODIS_Day_LST': 'LST_Day_1km',
        'MODIS_Terra_NDVI': 'NDVI',
        'MODIS_Terra_EVI': 'EVI',
        'MODIS_NDWI': ['sur_refl_b02', 'sur_refl_b06'],  # bands for NIR and SWIR, respectively
        'MODIS_NDVI': ['sur_refl_b02', 'sur_refl_b01'],  # bands for NIR and SWIR, respectively
        'MODIS_LAI': 'Lai',
        'MODIS_ET': 'ET',
        'TERRACLIMATE_ET': 'aet',  # unit in mm, monthly total
        'OpenET_ensemble': 'et_ensemble_mad',  # unit in mm, monthly total
        'Irrig_crop_OpenET_IrrMapper': 'et_ensemble_mad',  # unit in mm, monthly total
        'Irrig_crop_OpenET_LANID': 'et_ensemble_mad',  # unit in mm, monthly total
        'Rainfed_crop_OpenET_IrrMapper': 'et_ensemble_mad',  # unit in mm, monthly total
        'Rainfed_crop_OpenET_LANID': 'et_ensemble_mad',  # unit in mm, monthly total
        'TERRACLIMATE_RET': 'pet',
        'TERRACLIMATE_vap_pres': 'vap',
        'TERRACLIMATE_vap_pres_def': 'vpd',
        'GRIDMET_RET': 'eto',
        'GRIDMET_max_RH': 'rmax',
        'GRIDMET_min_RH': 'rmin',
        'GRIDMET_wind_vel': 'vs',
        'GRIDMET_short_rad': 'srad',
        'GRIDMET_vap_pres_def': 'vpd',
        'DAYMET_sun_hr': 'dayl',
        'USDA_CDL': 'cropland',
        'IrrMapper': 'classification',
        'LANID': None,  # The data holds annual datasets in separate band. Will process it out separately
        'AIM-HPA': None,  # The data holds annual datasets in separate band. Will process it out separately
        'Irrigation_Frac_IrrMapper': 'classification',
        'Irrigation_Frac_LANID': None,  # The data holds annual datasets in separate band. Will process it out separately
        'Rainfed_Frac_IrrMapper': 'classification',
        'Rainfed_Frac_LANID': None,  # The data holds annual datasets in separate band. Will process it out separately
        'Field_capacity': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Bulk_density': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Organic_carbon_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Sand_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'Clay_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
        'DEM': 'elevation',
        'Tree_cover': 'tree_canopy_cover'
    }

    gee_scale_dict = {
        'SMAP_SM': 1,
        'LANDSAT_NDWI': 1,
        'LANDSAT_NDVI': 1,
        'GPM_Precip': 1,
        'GRIDMET_Precip': 1,
        'MODIS_Day_LST': 0.02,
        'MODIS_Terra_NDVI': 0.0001,
        'MODIS_Terra_EVI': 0.0001,
        'MODIS_NDWI': 0.0001,
        'MODIS_NDVI': 0.0001,
        'MODIS_LAI': 0.1,
        'MODIS_ET': 0.1,
        'TERRACLIMATE_ET': 0.1,
        'OpenET_ensemble': 1,
        'Irrig_crop_OpenET_IrrMapper': 1,
        'Irrig_crop_OpenET_LANID': 1,
        'Rainfed_crop_OpenET_IrrMapper': 1,
        'Rainfed_crop_OpenET_LANID': 1,
        'TERRACLIMATE_RET': 0.1,
        'TERRACLIMATE_vap_pres': 0.001,
        'TERRACLIMATE_vap_pres_def': 0.01,
        'GRIDMET_RET': 1,
        'GRIDMET_max_RH': 1,
        'GRIDMET_min_RH': 1,
        'GRIDMET_wind_vel': 1,
        'GRIDMET_short_rad': 1,
        'GRIDMET_vap_pres_def': 1,
        'DAYMET_sun_hr': 1,
        'USDA_CDL': 1,
        'IrrMapper': 1,
        'LANID': 1,
        'AIM-HPA': 1,
        'Irrigation_Frac_IrrMapper': 1,
        'Irrigation_Frac_LANID': 1,
        'Rainfed_Frac_IrrMapper': 1,
        'Rainfed_Frac_LANID': 1,
        'Field_capacity': 1,
        'Bulk_density': 1,
        'Organic_carbon_content': 1,
        'Sand_content': 1,
        'Clay_content': 1,
        'DEM': 1,
        'Tree_cover': 1
    }

    aggregation_dict = {
        'SMAP_SM': ee.Reducer.sum(),
        'LANDSAT_NDWI': ee.Reducer.mean(),
        'LANDSAT_NDVI': ee.Reducer.mean(),
        'GPM_Precip': ee.Reducer.sum(),
        'GRIDMET_Precip': ee.Reducer.sum(),
        'MODIS_Day_LST': ee.Reducer.mean(),
        'MODIS_Terra_NDVI': ee.Reducer.mean(),
        'MODIS_Terra_EVI': ee.Reducer.mean(),
        'MODIS_NDWI': ee.Reducer.mean(),
        'MODIS_NDVI': ee.Reducer.mean(),
        'MODIS_LAI': ee.Reducer.mean(),
        'MODIS_ET': ee.Reducer.sum(),
        'TERRACLIMATE_ET': ee.Reducer.median(), # set for downloading monthly data, change if want to download yearly data
        'OpenET_ensemble': ee.Reducer.mean(),  # monthly data; doesn't matter whether use mean()or sum() as reducer. Change for yearly data download if needed.
        'Irrig_crop_OpenET_IrrMapper': ee.Reducer.sum(),
        'Irrig_crop_OpenET_LANID': ee.Reducer.sum(),  # as the data is downloaded at monthly resolution, setting mean/median/max as reducer won't make any difference. Setting it as sum() as it can be used for yearly aggregation
        'Rainfed_crop_OpenET_IrrMapper': ee.Reducer.sum(),
        'Rainfed_crop_OpenET_LANID': ee.Reducer.sum(),
        'TERRACLIMATE_RET': ee.Reducer.sum(),
        'TERRACLIMATE_vap_pres': ee.Reducer.mean(),
        'TERRACLIMATE_vap_pres_def': ee.Reducer.mean(),
        'GRIDMET_RET': ee.Reducer.sum(),
        'GRIDMET_max_RH': ee.Reducer.mean(),
        'GRIDMET_min_RH': ee.Reducer.mean(),
        'GRIDMET_wind_vel': ee.Reducer.mean(),
        'GRIDMET_short_rad': ee.Reducer.mean(),
        'GRIDMET_vap_pres_def': ee.Reducer.mean(),
        'DAYMET_sun_hr': ee.Reducer.mean(),
        'USDA_CDL': ee.Reducer.first(),
        'IrrMapper': ee.Reducer.max(),
        'LANID': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': ee.Reducer.max(),
        'Irrigation_Frac_LANID': None,
        'Rainfed_Frac_IrrMapper': ee.Reducer.max(),
        'Rainfed_Frac_LANID': ee.Reducer.mean(),
        'Field_capacity': ee.Reducer.mean(),
        'Bulk_density': ee.Reducer.mean(),
        'Organic_carbon_content': ee.Reducer.mean(),
        'Sand_content': ee.Reducer.mean(),
        'Clay_content': ee.Reducer.mean(),
        'DEM': None,
        'Tree_cover': ee.Reducer.mean()
    }

    # # Note on start date and end date dictionaries
    # The start and end dates have been set based on what duration of data can be downloaded.
    # They may not exactly match with the data availability in GEE
    # In most cases the end date is shifted a month later to cover the end month's data

    month_start_date_dict = {
        'SMAP_SM': datetime(2015, 4, 1),
        'LANDSAT_NDWI': datetime(2013, 4, 1),
        'LANDSAT_NDVI': datetime(2013, 4, 1),
        'GPM_Precip': datetime(2006, 6, 1),
        'GRIDMET_Precip': datetime(1979, 1, 1),
        'MODIS_Day_LST': datetime(2000, 2, 1),
        'MODIS_Terra_NDVI': datetime(2000, 2, 1),
        'MODIS_Terra_EVI': datetime(2000, 2, 1),
        'MODIS_NDWI': datetime(2000, 2, 1),
        'MODIS_NDVI': datetime(2000, 2, 1),
        'MODIS_LAI': datetime(2002, 7, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'TERRACLIMATE_ET': datetime(1958, 1, 1),
        'OpenET_ensemble': datetime(2016, 1, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(2016, 1, 1),
        'Irrig_crop_OpenET_LANID': datetime(2016, 1, 1),
        'Rainfed_crop_OpenET_IrrMapper': datetime(2016, 1, 1),
        'Rainfed_crop_OpenET_LANID': datetime(2016, 1, 1),
        'TERRACLIMATE_RET': datetime(1958, 1, 1),
        'TERRACLIMATE_vap_pres': datetime(1958, 1, 1),
        'TERRACLIMATE_vap_pres_def': datetime(1958, 1, 1),
        'GRIDMET_RET': datetime(1979, 1, 1),
        'GRIDMET_max_RH': datetime(1979, 1, 1),
        'GRIDMET_min_RH': datetime(1979, 1, 1),
        'GRIDMET_wind_vel': datetime(1979, 1, 1),
        'GRIDMET_short_rad': datetime(1979, 1, 1),
        'GRIDMET_vap_pres_def': datetime(1979, 1, 1),
        'DAYMET_sun_hr': datetime(1980, 1, 1),
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
        'IrrMapper': datetime(1986, 1, 1),
        'LANID': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(1986, 1, 1),
        'Irrigation_Frac_LANID': None,
        'Rainfed_Frac_IrrMapper': datetime(1986, 1, 1),
        'Rainfed_Frac_LANID': None,
        'Field_capacity': None,
        'Bulk_density': None,
        'Organic_carbon_content': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2000, 1, 1)
    }

    month_end_date_dict = {
        'SMAP_SM': datetime(2022, 8, 2),
        'LANDSAT_NDWI': datetime(2022, 1, 1),
        'LANDSAT_NDVI': datetime(2022, 1, 1),
        'GPM_Precip': datetime(2021, 9, 1),
        'GRIDMET_Precip': datetime(2023, 9, 15),
        'MODIS_Day_LST': datetime(2023, 8, 29),
        'MODIS_Terra_NDVI': datetime(2023, 8, 13),
        'MODIS_Terra_EVI': datetime(2023, 8, 13),
        'MODIS_NDWI': datetime(2023, 8, 29),
        'MODIS_NDVI': datetime(2023, 8, 29),
        'MODIS_LAI': datetime(2023, 9, 6),
        'MODIS_ET': datetime(2023, 8, 29),
        'TERRACLIMATE_ET': datetime(2022, 12, 1),
        'OpenET_ensemble': datetime(2022, 12, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(2021, 1, 1),
        'Irrig_crop_OpenET_LANID': datetime(2021, 1, 1),
        'Rainfed_crop_OpenET_IrrMapper': datetime(2021, 1, 1),
        'Rainfed_crop_OpenET_LANID': datetime(2021, 1, 1),
        'TERRACLIMATE_RET': datetime(2022, 12, 31),
        'TERRACLIMATE_vap_pres': datetime(2022, 12, 31),
        'TERRACLIMATE_vap_pres_def': datetime(2022, 12, 31),
        'GRIDMET_RET': datetime(2022, 12, 1),
        'GRIDMET_max_RH': datetime(2022, 12, 1),
        'GRIDMET_min_RH': datetime(2022, 12, 1),
        'GRIDMET_wind_vel': datetime(2022, 12, 1),
        'GRIDMET_short_rad': datetime(2022, 12, 1),
        'GRIDMET_vap_pres_def': datetime(2022, 12, 1),
        'DAYMET_sun_hr': datetime(2022, 12, 31),
        'USDA_CDL': datetime(2022, 1, 1),
        'IrrMapper': datetime(2020, 1, 1),
        'LANID': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(2020, 1, 1),
        'Irrigation_Frac_LANID': None,
        'Rainfed_Frac_IrrMapper': datetime(2020, 1, 1),
        'Rainfed_Frac_LANID': None,
        'Field_capacity': None,
        'Bulk_density': None,
        'Organic_carbon_content': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2015, 1, 1)
    }

    year_start_date_dict = {
        'SMAP_SM': datetime(2015, 1, 1),
        'LANDSAT_NDWI': datetime(2013, 1, 1),
        'LANDSAT_NDVI': datetime(2013, 1, 1),
        'GPM_Precip': datetime(2006, 1, 1),
        'GRIDMET_Precip': datetime(1979, 1, 1),
        'MODIS_Day_LST': datetime(2000, 1, 1),
        'MODIS_Terra_NDVI': datetime(2000, 1, 1),
        'MODIS_Terra_EVI': datetime(2000, 1, 1),
        'MODIS_NDWI': datetime(2000, 1, 1),
        'MODIS_NDVI': datetime(2000, 1, 1),
        'MODIS_LAI': datetime(2002, 1, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'TERRACLIMATE_ET': datetime(1958, 1, 1),
        'OpenET_ensemble': datetime(2016, 1, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(2016, 1, 1),
        'Irrig_crop_OpenET_LANID': datetime(2016, 1, 1),
        'Rainfed_crop_OpenET_IrrMapper': datetime(2016, 1, 1),
        'Rainfed_crop_OpenET_LANID': datetime(2016, 1, 1),
        'TERRACLIMATE_RET': datetime(1958, 1, 1),
        'TERRACLIMATE_vap_pres': datetime(1958, 1, 1),
        'TERRACLIMATE_vap_pres_def': datetime(1958, 1, 1),
        'GRIDMET_RET': datetime(1979, 1, 1),
        'GRIDMET_max_RH': datetime(1979, 1, 1),
        'GRIDMET_min_RH': datetime(1979, 1, 1),
        'GRIDMET_wind_vel': datetime(1979, 1, 1),
        'GRIDMET_short_rad': datetime(1979, 1, 1),
        'GRIDMET_vap_pres_def': datetime(1979, 1, 1),
        'DAYMET_sun_hr': datetime(1980, 1, 1),
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
        'IrrMapper': datetime(1986, 1, 1),
        'LANID': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(1986, 1, 1),
        'Irrigation_Frac_LANID': None,
        'Rainfed_Frac_IrrMapper': datetime(1986, 1, 1),
        'Rainfed_Frac_LANID': None,
        'Field_capacity': None,
        'Bulk_density': None,
        'Organic_carbon_content': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2000, 1, 1)
    }

    year_end_date_dict = {
        'SMAP_SM': datetime(2023, 1, 1),
        'LANDSAT_NDWI': datetime(2022, 1, 1),
        'LANDSAT_NDVI': datetime(2022, 1, 1),
        'GPM_Precip': datetime(2022, 1, 1),
        'GRIDMET_Precip': datetime(2024, 1, 1),
        'MODIS_Day_LST': datetime(2024, 1, 1),
        'MODIS_Terra_NDVI': datetime(2024, 1, 1),
        'MODIS_Terra_EVI': datetime(2024, 1, 1),
        'MODIS_NDWI': datetime(2024, 1, 1),
        'MODIS_NDVI': datetime(2024, 1, 1),
        'MODIS_LAI': datetime(2024, 1, 1),
        'MODIS_ET': datetime(2024, 1, 1),
        'TERRACLIMATE_ET': datetime(2023, 1, 1),
        'OpenET_ensemble': datetime(2023, 1, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(2021, 1, 1),
        'Irrig_crop_OpenET_LANID': datetime(2021, 1, 1),
        'Rainfed_crop_OpenET_IrrMapper': datetime(2021, 1, 1),
        'Rainfed_crop_OpenET_LANID': datetime(2021, 1, 1),
        'TERRACLIMATE_RET': datetime(2023, 1, 1),
        'TERRACLIMATE_vap_pres': datetime(2023, 1, 1),
        'TERRACLIMATE_vap_pres_def': datetime(2023, 1, 1),
        'GRIDMET_RET': datetime(2024, 12, 1),
        'GRIDMET_max_RH': datetime(2024, 1, 1),
        'GRIDMET_min_RH': datetime(2024, 1, 1),
        'GRIDMET_wind_vel': datetime(2024, 1, 1),
        'GRIDMET_short_rad': datetime(2024, 1, 1),
        'GRIDMET_vap_pres_def': datetime(2024, 12, 1),
        'DAYMET_sun_hr': datetime(2023, 1, 1),
        'USDA_CDL': datetime(2022, 1, 1),
        'IrrMapper': datetime(2021, 1, 1),
        'LANID': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(2021, 1, 1),
        'Irrigation_Frac_LANID': None,
        'Rainfed_Frac_IrrMapper': datetime(2021, 1, 1),
        'Rainfed_Frac_LANID': None,
        'Field_capacity': None,
        'Bulk_density': None,
        'Organic_carbon_content': None,
        'Sand_content': None,
        'Clay_content': None,
        'DEM': None,
        'Tree_cover': datetime(2015, 1, 1)
    }

    return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name], \
           month_start_date_dict[data_name], month_end_date_dict[data_name], year_start_date_dict[data_name], \
           year_end_date_dict[data_name],


def cloud_cover_filter(data_name, start_date, end_date, from_bit, to_bit, geometry_bounds):
    """
    Applies cloud cover mask on GEE data.

    :param data_name: Data Name.
           Valid dataset include- ['MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_NDVI'].
    :param start_date: Start date of data to download. Generated from download_gee_data() func.
    :param end_date: End date of data to download. Generated from download_gee_data() func.
    :param from_bit: Start bit to consider for masking.
    :param to_bit: End bit to consider for masking.
    :param geometry_bounds: GEE geometry object.

    :return: Cloud filtered imagecollection.
    """

    def bitwise_extract(img):
        """
        Applies cloudmask on image.
        :param img: The image.

        :return Cloud-masked image.
        """
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        global qc_img
        if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
            qc_img = img.select('DetailedQA')
        elif data_name in ['MODIS_NDWI', 'MODIS_NDVI']:
            qc_img = img.select('StateQA')

        masksize = ee.Number(1).add(to_bit).subtract(from_bit)
        mask = ee.Number(1).leftShift(masksize).subtract(1)
        apply_mask = qc_img.rightShift(from_bit).bitwiseAnd(mask).lte(1)
        return img.updateMask(apply_mask)

    if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
        # filterBounds are not necessary, added it to reduce processing extent
        image = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked

    elif data_name in ['MODIS_NDWI', 'MODIS_NDVI']:
        image = ee.ImageCollection('MODIS/061/MOD09A1').filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked


# # The download_gee_data_for_grow_season() function isn't fully optimized. Might need to change things at the
# #  get_gee_dict() function, Not using this function for the current project
# def download_gee_data_for_grow_season \
#                 (data_name, download_dir, year_list, merge_keyword, grid_shape_large, refraster_westUS=WestUS_raster):
#     """
#     process and download data from GEE based on varying growing season. This code will mosaic data for different growing
#     season together. So there will be visible difference in data. Consider smoothening the grids used for downloading.
#
#     **Saving as a reference code. Will not use in this project. Rather data will be downloaded for each month
#     (for selected years) and then merged together based on growing season to make the data smoother.
#
#     :param data_name: Data name.
#     Current valid data names are -
#         ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_Precip', 'GRIDMET_Precip',
#         'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
#         'MODIS_ET']
#     :param download_dir: File path of download directory.
#     :param year_list: List of years to download data for, e.g. [2000, 2005, 2010, 2015].
#     :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
#     :param grid_shape_large: File path of grid shape for which data will be downloaded and mosaiced.
#     :param refraster_westUS: Reference raster to use for merging downloaded datasets.
#
#     :return: File path of downloaded raster data.
#     """
#     ee.Initialize()
#     makedirs([download_dir])
#
#     downloaded_raster_dict = {}
#
#     grow_season_dict = {'Apr01-Oct31': [4, 10], 'Jan01-Dec31': [1, 12], 'Jul01-Aug31': [7, 8], 'Mar01-Nov30': [3, 11],
#                         'May01-Sep30': [5, 9]}  # A dictionary for assigning growing season months
#
#     for year in year_list:
#         grids = gpd.read_file(grid_shape_large)
#         grids = grids.sort_values(by='grid_no', ascending=True)
#         growing_season = grids['GrowSeason']
#         grid_geometry = grids['geometry']
#         grid_no = grids['grid_no']
#
#         start_year, end_year, start_month, end_month = None, None, None, None
#
#         for grid, grow_period, serial in zip(grid_geometry, growing_season, grid_no):
#             roi = grid.bounds
#             gee_extent = ee.Geometry.Rectangle(roi)
#
#             # Selecting month ranges based on grid files' growing season periods
#             if grow_period == 'Apr01-Oct31':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['Apr01-Oct31'][0], grow_season_dict['Apr01-Oct31'][1]
#
#             elif grow_period == 'Jan01-Dec31':
#                 start_year = year
#                 end_year = year + 1
#                 start_month, end_month = grow_season_dict['Jan01-Dec31'][0], 1  # up to January 1st of next year
#
#             elif grow_period == 'Jul01-Aug31':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['Jul01-Aug31'][0], grow_season_dict['Jul01-Aug31'][1]
#
#             elif grow_period == 'Mar01-Nov30':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['Mar01-Nov30'][0], grow_season_dict['Mar01-Nov30'][1]
#
#             elif grow_period == 'May01-Sep30':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['May01-Sep30'][0], grow_season_dict['May01-Sep30'][1]
#
#             start_date = ee.Date.fromYMD(start_year, start_month, 1)
#             end_date = ee.Date.fromYMD(end_year, end_month + 1, 1)
#             data, band, multiply_scale, reducer, month_start_range, month_end_range, \
#             year_start_range, year_end_range = get_gee_dict(data_name)
#
#             if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
#                 download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band). \
#                     reduce(reducer).multiply(multiply_scale).toFloat()
#
#             elif data_name == 'MODIS_NDWI':
#                 nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]) \
#                     .reduce(reducer).multiply(multiply_scale).toFloat()
#                 swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]) \
#                     .reduce(reducer).multiply(multiply_scale).toFloat()
#                 download_data = nir.subtract(swir).divide(nir.add(swir))
#
#             else:
#                 download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
#                     filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()
#
#             data_url = download_data.getDownloadURL({'name': data_name,
#                                                      'crs': 'EPSG:4269',  # NAD83
#                                                      'scale': 2200,  # in meter. equal to ~0.02 deg
#                                                      'region': gee_extent,
#                                                      'format': 'GEO_TIFF'})
#
#             key_word = data_name
#             local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(serial)}.tif')
#             print('Downloading', local_file_name, '.....')
#             r = requests.get(data_url, allow_redirects=True)
#             open(local_file_name, 'wb').write(r.content)
#
#         mosaic_name = f'{data_name}_{year}.tif'
#         mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
#         makedirs([mosaic_dir])
#         downloaded_arr, downloaded_raster = mosaic_rasters_from_directory(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster_westUS,
#                                                            search_by=f'*{year}*.tif', nodata=no_data_value)
#         print('Downloaded Data Merged')
#         downloaded_raster_dict[mosaic_name] = downloaded_raster
#
#     return downloaded_raster_dict

def download_soil_datasets(data_name, download_dir, merge_keyword, grid_shape, refraster_westUS=WestUS_raster,
                           refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape):
    """
    Download soil datasets from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['Field_capacity', 'Bulk_density', 'Organic_carbon_content', 'Sand_content','Clay_content']
    :param download_dir: File path of download directory.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    ee.Initialize()

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    if data_name in ['Field_capacity', 'Bulk_density', 'Organic_carbon_content', 'Sand_content', 'Clay_content']:
        data, all_bands, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

        # selecting datasets with all bands ['b0', 'b10', 'b30', 'b60', 'b100', 'b200']
        data_all_bands = ee.Image(data).select(all_bands)

        # calculating band average
        dataset_mean = data_all_bands.reduce(reducer)

        # Loading grid files to be used for data download
        grids = gpd.read_file(grid_shape)
        grids = grids.sort_values(by='grid_no', ascending=True)
        grid_geometry = grids['geometry']
        grid_no = grids['grid_no']

        for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
            roi = geometry.bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            # making data url
            data_url = dataset_mean.getDownloadURL({'name': data_name,
                                                    'crs': 'EPSG:4269',  # NAD83
                                                    'scale': 2200,  # in meter. equal to ~0.02 deg
                                                    'region': gee_extent,
                                                    'format': 'GEO_TIFF'})
            key_word = data_name
            local_file_name = os.path.join(download_dir, f'{key_word}_{str(grid_sr)}.tif')
            print('Downloading', local_file_name, '.....')
            r = requests.get(data_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)

        mosaic_name = f'{data_name}.tif'
        mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
        clip_dir = os.path.join(download_dir, f'{merge_keyword}')

        makedirs([clip_dir, mosaic_dir])
        merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                                  raster_name=mosaic_name,
                                                                  ref_raster=refraster_gee_merge,
                                                                  search_by=f'*.tif', nodata=no_data_value)

        clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                       output_raster_dir=clip_dir, clip_and_resample=True,
                                       use_ref_width_height=False, resolution=model_res,
                                       ref_raster=refraster_westUS)

        print(f'{data_name} data downloaded and merged')

    else:
        pass


def download_tree_cover_data(data_name, download_dir, merge_keyword, grid_shape, refraster_westUS=WestUS_raster,
                             refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape):
    """
    Download Tree Cover data from GEE.

    :param data_name: Data name. Use 'Tree_cover'.
    :param download_dir: File path of download directory.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.
    :param terrain_slope : If slope data download is needed in degrees from GEE directly. Defaults to False to download
                           DEM data only. The DEM data will be later processed to 'percent' slope data using gdal.

    :return: None.
    """
    ee.Initialize()

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # getting dataset information from the data dictionary
    data, band, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

    # filtering data
    dataset = ee.ImageCollection(data).filter(ee.Filter.date('2000-01-01', '2015-01-01')).select(band) \
        .reduce(reducer).toFloat()

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
        roi = geometry.bounds
        gee_extent = ee.Geometry.Rectangle(roi)

        # making data url
        data_url = dataset.getDownloadURL({'name': data_name,
                                           'crs': 'EPSG:4269',  # NAD83
                                           'scale': 2200,  # in meter. equal to ~0.02 deg
                                           'region': gee_extent,
                                           'format': 'GEO_TIFF'})
        key_word = data_name
        local_file_name = os.path.join(download_dir, f'{key_word}_{str(grid_sr)}.tif')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

    mosaic_name = f'{data_name}.tif'
    mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
    clip_dir = os.path.join(download_dir, f'{merge_keyword}')

    makedirs([clip_dir, mosaic_dir])
    merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                              raster_name=mosaic_name,
                                                              ref_raster=refraster_gee_merge,
                                                              search_by=f'*.tif', nodata=no_data_value)

    clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                   output_raster_dir=clip_dir, clip_and_resample=True,
                                   use_ref_width_height=False, resolution=model_res,
                                   ref_raster=refraster_westUS)

    print(f'{data_name} data downloaded and merged')


def download_DEM_Slope_data(data_name, download_dir, merge_keyword, grid_shape, refraster_westUS=WestUS_raster,
                            refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape,
                            terrain_slope=False):
    """
    Download DEM/Slope data from GEE.

    :param data_name: Data name. Use keyword 'DEM' for downloading both DEM and Slope data. SLope is downloaded
                      in degrees.
    :param download_dir: File path of download directory.
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.
    :param terrain_slope : If slope data download is needed in degrees from GEE directly. Defaults to False to download
                           DEM data only. The DEM data will be later processed to 'percent' slope data using gdal.

    :return: None.
    """
    ee.Initialize()

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # getting dataset information from the data dictionary
    data, band, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

    # filtering data
    dataset = ee.Image(data).select(band).multiply(multiply_scale).toFloat()

    if terrain_slope:
        dataset = ee.Terrain.slope(dataset)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
        roi = geometry.bounds
        gee_extent = ee.Geometry.Rectangle(roi)

        # making data url
        data_url = dataset.getDownloadURL({'name': data_name,
                                           'crs': 'EPSG:4269',  # NAD83
                                           'scale': 2200,  # in meter. equal to ~0.02 deg
                                           'region': gee_extent,
                                           'format': 'GEO_TIFF'})
        key_word = data_name
        local_file_name = os.path.join(download_dir, f'{key_word}_{str(grid_sr)}.tif')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

    mosaic_name = f'{data_name}.tif'
    mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
    clip_dir = os.path.join(download_dir, f'{merge_keyword}')

    makedirs([clip_dir, mosaic_dir])
    merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                              raster_name=mosaic_name,
                                                              ref_raster=refraster_gee_merge,
                                                              search_by=f'*.tif', nodata=no_data_value)

    clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                   output_raster_dir=clip_dir, clip_and_resample=True,
                                   use_ref_width_height=False, resolution=model_res,
                                   ref_raster=refraster_westUS)

    print(f'{data_name} data downloaded and merged')


# # The download_gee_data_yearly() function isn't fully optimized. Might need to change things at the
# #  get_gee_dict() function. ALso, All it might need modification to download all datasets available in
# the get_gee_dict() function. Not using this function for the current project
def download_gee_data_yearly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
                             refraster_westUS=WestUS_raster, refraster_gee_merge=GEE_merging_refraster_large_grids,
                             westUS_shape=WestUS_shape):
    """
    Download data (at yearly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_Precip', 'GRIDMET_Precip',
        'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_NDVI', 'MODIS_LAI', 'MODIS_ET',
        'TERRACLIMATE_RET', 'TERRACLIMATE_vap_pres', 'TERRACLIMATE_vap_pres_def', 'GRIDMET_max_RH',
        'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'GRIDMET_RET', 'GRIDMET_vap_pres_def',
        'DAYMET_sun_hr']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 4-12 use (4, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years
        start_date = ee.Date.fromYMD(year, month_range[0], 1)
        start_date_dt = datetime(year, month_range[0], 1)
        if month_range[1] < 12:
            end_date = ee.Date.fromYMD(year, month_range[1] + 1, 1)
            end_date_dt = datetime(year, month_range[1] + 1, 1)
        else:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
            end_date_dt = datetime(year + 1, 1, 1)

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) & (end_date_dt <= year_end_range):

            for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
                roi = geometry.bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Filtering/processing datasets with data ranges, cloudcover, geometry, band, reducer, scale
                if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                    download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent). \
                        select(band).reduce(reducer).multiply(multiply_scale).toFloat()

                elif data_name == 'MODIS_NDWI':
                    nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    download_data = nir.subtract(swir).divide(nir.add(swir))

                elif data_name == 'MODIS_NDVI':
                    nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    red = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    download_data = nir.subtract(red).divide(nir.add(red))

                elif data_name == 'USDA_CDL':
                    cdl_dataset = ee.ImageCollection(data).filter((ee.Filter.calendarRange(year, year, 'year'))) \
                        .select(band).reduce(reducer).multiply(multiply_scale).toFloat()

                    # List of non-crop pixels
                    noncrop_list = ee.List([60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 112, 121, 122, 123,
                                            124, 131, 141, 142, 143, 152, 190,
                                            195])  # 176 (pasture) kept in downloaded data

                    # Filtering out non-crop pixels. In non-crop pixels, assigning 0 and in crop pixels assigning 1
                    cdl_mask = cdl_dataset.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

                    # Masking with cdl mask to assign nodata value on non crop pixels
                    cdl_cropland = cdl_dataset.updateMask(cdl_mask)
                    download_data = cdl_cropland

                elif data_name == 'GRIDMET_RET':
                    # multiplying by 0.85 to applying bias correction in GRIDMET RET. GRIDMET RET is overestimated
                    # by 12-31% across CONUS (Blankenau et al. (2020). Senay et al. (2022) applied 0.85 as constant
                    # bias correction factor.
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(0.85).multiply(multiply_scale).toFloat()

                elif data_name == 'DAYMET_sun_hr':
                    # dividing by 3600 to convert from second to hr
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).divide(3600).multiply(multiply_scale).toFloat()

                else:
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                data_url = download_data.getDownloadURL({'name': data_name,
                                                         'crs': 'EPSG:4269',  # NAD83
                                                         'scale': 2200,  # in meter. equal to ~0.02 deg
                                                         'region': gee_extent,
                                                         'format': 'GEO_TIFF'})
                key_word = data_name
                local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')
                print('Downloading', local_file_name, '.....')
                r = requests.get(data_url, allow_redirects=True)
                open(local_file_name, 'wb').write(r.content)

                # # Joining LANID and AIM-HPA
                # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
                # Remapping the irrigated 0 values to 1 and then applying a mask
                try:
                    arr = read_raster_arr_object(local_file_name, get_file=False)

                except:
                    print(f'Downloaded data corrupted. Re-downloading {local_file_name}.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(local_file_name, 'wb').write(r.content)

            mosaic_name = f'{data_name}_{year}.tif'
            mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
            clip_dir = os.path.join(download_dir, f'{merge_keyword}')

            makedirs([clip_dir, mosaic_dir])
            merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir, output_dir=mosaic_dir,
                                                                      raster_name=mosaic_name,
                                                                      ref_raster=refraster_gee_merge,
                                                                      search_by=f'*{year}*.tif', nodata=no_data_value)

            clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                           output_raster_dir=clip_dir, clip_and_resample=True,
                                           use_ref_width_height=False, resolution=model_res,
                                           ref_raster=refraster_westUS)

            print(f'{data_name} yearly data downloaded and merged')

        else:
            pass


def download_gee_data_monthly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
                              use_cpu_while_multidownloading=15, refraster_westUS=WestUS_raster,
                              refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape):
    """
    Download data (at monthly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_Precip', 'GRIDMET_Precip',
        'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_NDVI', 'MODIS_LAI', 'MODIS_ET',
        'TERRSCLIMATE_ET', 'TERRACLIMATE_RET', 'TERRACLIMATE_vap_pres', 'TERRACLIMATE_vap_pres_def', 'GRIDMET_max_RH',
        'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'GRIDMET_RET', 'GRIDMET_vap_pres_def',
        'DAYMET_sun_hr']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

    # the variables in data_exclude_list can't be downloaded by this function. They have separate download functions
    data_exclude_list = ['Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID',
                         'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID',
                         'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID',
                         'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID', 'USDA_CDL',
                         'Field_capacity', 'Bulk_density', 'Organic_carbon_content',
                         'Sand_content', 'Clay_content', 'DEM', 'Effect_precip_DK', 'Tree_cover']

    if data_name not in data_exclude_list:
        for year in year_list:  # first loop for years
            for month in month_list:  # second loop for months
                print('********')
                print(f'Getting data urls for year={year}, month={month}.....')

                # Setting date ranges
                start_date = ee.Date.fromYMD(year, month, 1)
                start_date_dt = datetime(year, month, 1)

                if month < 12:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
                    end_date_dt = datetime(year, month + 1, 1)

                else:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                    end_date_dt = datetime(year + 1, 1, 1)

                # a condition to check whether start and end date falls in the available data range in GEE
                # if not the block will not be executed
                if (start_date_dt >= month_start_range) & (end_date_dt <= month_end_range):
                    # will collect url and file name in url list and local_file_paths_list
                    data_url_list = []
                    local_file_paths_list = []

                    for i in range(len(grid_no)):  # third loop for grids
                        # converting grid geometry info to a GEE extent
                        grid_sr = grid_no[i]
                        roi = grid_geometry[i].bounds
                        gee_extent = ee.Geometry.Rectangle(roi)

                        # Filtering/processing datasets with data ranges, cloudcover, geometry, band, reducer, scale
                        if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                            download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1,
                                                               gee_extent).select(band). \
                                                                reduce(reducer).multiply(multiply_scale).toFloat()

                        elif data_name == 'MODIS_NDWI':
                            nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]). \
                                reduce(reducer).multiply(multiply_scale).toFloat()
                            swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(
                                band[1]). \
                                reduce(reducer).multiply(multiply_scale).toFloat()
                            download_data = nir.subtract(swir).divide(nir.add(swir))

                        elif data_name == 'MODIS_NDVI':
                            nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]). \
                                reduce(reducer).multiply(multiply_scale).toFloat()
                            red = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]). \
                                reduce(reducer).multiply(multiply_scale).toFloat()
                            download_data = nir.subtract(red).divide(nir.add(red))

                        elif data_name in ('TERRACLIMATE_ET', 'OpenET_ensemble'):
                            download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                                filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                        elif data_name == 'GRIDMET_RET':
                            # multiplying by 0.85 to applying bias correction in GRIDMET RET. GRIDMET RET is overestimated
                            # by 12-31% across CONUS (Blankenau et al. (2020). Senay et al. (2022) applied 0.85 as constant
                            # bias correction factor.
                            download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                                filterBounds(gee_extent).reduce(reducer).multiply(0.85).multiply(
                                multiply_scale).toFloat()

                        elif data_name == 'DAYMET_sun_hr':
                            # dividing by 3600 to convert from second to hr
                            download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                                filterBounds(gee_extent).reduce(reducer).divide(3600).multiply(multiply_scale).toFloat()

                        else:
                            download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                                filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                        # Getting Data URl for each grid from GEE
                        # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to
                        # retry failed connections
                        try:
                            data_url = download_data.getDownloadURL({'name': data_name,
                                                                     'crs': 'EPSG:4269',  # NAD83
                                                                     'scale': 2200,  # in meter. equal to ~0.02 deg
                                                                     'region': gee_extent,
                                                                     'format': 'GEO_TIFF'})
                        except:
                            data_url = download_data.getDownloadURL({'name': data_name,
                                                                     'crs': 'EPSG:4269',  # NAD83
                                                                     'scale': 2200,  # in meter. equal to ~0.02 deg
                                                                     'region': gee_extent,
                                                                     'format': 'GEO_TIFF'})

                        key_word = data_name
                        local_file_path = os.path.join(download_dir,
                                                       f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

                        # Appending data url and local file path (to save data) to a central list
                        data_url_list.append(data_url)
                        local_file_paths_list.append(local_file_path)

                        # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                        # there is enough data url gathered for download.
                        if (len(data_url_list) == 120) | (
                                i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                            # Combining url and file paths together to pass in multiprocessing
                            urls_to_file_paths_compile = []
                            for j, k in zip(data_url_list, local_file_paths_list):
                                urls_to_file_paths_compile.append([j, k])

                            # Download data by multi-processing/multi-threading
                            download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                                   use_cpu=use_cpu_while_multidownloading)

                            # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                            # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                            data_url_list = []
                            local_file_paths_list = []

                    mosaic_name = f'{data_name}_{year}_{month}.tif'
                    mosaic_dir = os.path.join(download_dir, f'{merge_keyword}', 'merged')
                    clip_dir = os.path.join(download_dir, f'{merge_keyword}')

                    makedirs([clip_dir, mosaic_dir])
                    search_by = f'*{year}_{month}*.tif'
                    merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir,
                                                                              output_dir=mosaic_dir,
                                                                              raster_name=mosaic_name,
                                                                              ref_raster=refraster_gee_merge,
                                                                              search_by=search_by,
                                                                              nodata=no_data_value)

                    clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                                   output_raster_dir=clip_dir, clip_and_resample=True,
                                                   use_ref_width_height=False, resolution=model_res,
                                                   ref_raster=refraster_westUS)

                    print(f'{data_name} monthly data downloaded and merged')

                else:
                    print(f'Data for year {year}, month {month} is out of range. Skipping query')
                    pass


def get_data_GEE_saveTopath(url_and_file_path):
    """
    Uses data url to get data from GEE and save it to provided local file paths.

    :param url_and_file_path: A list of tuples where each tuple has the data url (1st member) and local file path
                             (2nd member).
    :return: None
    """
    # unpacking tuple
    data_url, file_path = url_and_file_path

    # get data from GEE
    r = requests.get(data_url, allow_redirects=True)
    print('Downloading', file_path, '.....')

    # save data to local file path
    open(file_path, 'wb').write(r.content)

    # This is a check block to see if downloaded datasets are OK
    # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
    # Re-downloading the same data might not have that error
    try:
        arr = read_raster_arr_object(file_path, get_file=False)

    except:
        print(f'Downloaded data corrupted. Re-downloading {file_path}.....')
        r = requests.get(data_url, allow_redirects=True)
        open(file_path, 'wb').write(r.content)


def download_data_from_GEE_by_multiprocess(download_urls_fp_list, use_cpu=2):
    """
    Use python multiprocessing library to download data from GEE in a multi-thread approach. This function is a
    wrapper over get_data_GEE_saveTopath() function providing muti-threading support.

    :param download_urls_fp_list: A list of tuples where each tuple has the data url (1st member) and local file path
                                  (2nd member).
    :param use_cpu: Number of CPU/core (Int) to use for downloading. Default set to 15.

    :return: None.
    """
    # Using ThreadPool() instead of pool() as this is an I/O bound job not CPU bound
    # Using imap() as it completes assigns one task at a time to the Thread
    # Pool() and blocks until each task is complet
    print('######')
    print('Downloading data from GEE..')
    print(f'{cpu_count()} CPUs on this machine. Engaging {use_cpu} CPUs for downloading')
    print('######')

    pool = ThreadPool(use_cpu)
    results = pool.imap(get_data_GEE_saveTopath, download_urls_fp_list)
    pool.close()
    pool.join()


def download_Irr_frac_from_IrrMapper_yearly(data_name, download_dir, year_list, grid_shape,
                                            use_cpu_while_multidownloading=15):
    """
    Download IrrMapper Irrigated fraction data (at 2km scale) at yearly scale from GEE for 11 states in the Western US
    WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM for 2000-2020.
    *** For downloading irrigated fraction data for ND, SD, OK, KS, NE, and TX use download_Irr_frac_from_LANID_yearly()
    function.

    ########################
    # READ ME (for Irrigation Data)
    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020,
    whereas LANID data consists datasets of these 11 states and ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017.
    Our target year is 2000-2020. As LANID data isn't available for all the years we are using IrrMapper data as
    irrigated cropland data for 2000-2020 for the available 11 states' data (also it's available in GEE already).
    For the other 06 states, we clipped LANID data and compiled them as a GEE asset for 2000-2020 (2018-2020 data
    considered the same as 2017 data to fill the missing values). In summary, for WA, OR, CA, ID, NV, UT, AZ, MT, WY,
    CO, and NM (11 states) the irrigated field data comes from IrrMapper and for ND, SD, OK, KS, NE, and TX the
    irrigated field data comes from LANID.
    ########################
.
    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrigation_Frac_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, _, _, year_start_range, year_end_range = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years
        print('********')
        print(f'Getting data urls for year={year}.....')
        start_date_dt = datetime(year, 1, 1)
        end_date_dt = datetime(year, 12, 31)

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) & (end_date_dt <= year_end_range):
            # Filtering data for the year range and reducing data
            irrmap_imcol = ee.ImageCollection(data)
            irrmap = irrmap_imcol.filter(ee.Filter.calendarRange(year, year, 'year')).select(band).reduce(reducer)

            # IrrMapper projection extraction
            projection_irrmap = ee.Image(irrmap_imcol.first()).projection()
            projection2km_scale = projection_irrmap.atScale(2200)

            # In IrrMapper dataset irrigated fields are assigned as 0
            # Converting the irrigated values to 1 and setting others as null
            mask = irrmap.eq(0)
            irr_mask_only = irrmap.updateMask(mask).remap([0], [1]).setDefaultProjection(crs=projection_irrmap)

            # 30m Irrigation pixel count in each 2km pixel
            irr_pixel_count = irr_mask_only.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
                .reproject(crs=projection2km_scale)
            # In IrrMapper dataset irrigated fields are assigned as 0
            # Converting the irrigated values to 1 and setting others as 0
            irr_mask_with_total = irrmap.eq(0).setDefaultProjection(crs=projection_irrmap)

            # Total number of 30m pixels count in each 2km pixel
            total_pixel_count = irr_mask_with_total.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
                .reproject(crs=projection2km_scale)

            # counting fraction of irrigated lands in a pixel
            irrig_frac = irr_pixel_count.divide(total_pixel_count)

            # second loop for grids
            data_url_list = []
            local_file_paths_list = []

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                try:
                    data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                          'crs': 'EPSG:4269',  # NAD83
                                                          'scale': 2200,  # in meter
                                                          'region': gee_extent,
                                                          'format': 'GEO_TIFF'})

                except:
                    data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                          'crs': 'EPSG:4269',  # NAD83
                                                          'scale': 2200,  # in meter
                                                          'region': gee_extent,
                                                          'format': 'GEO_TIFF'})

                key_word = data_name
                local_file_path = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for j, k in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([j, k])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

        else:
            pass


def download_Irr_frac_from_LANID_yearly(data_name, download_dir, year_list, grid_shape,
                                        use_cpu_while_multidownloading=2):
    """
    Download LANID Irrigated fraction data (at 2km scale) at yearly scale from GEE for 6 states in the Western US
    ND, SD, OK, KS, NE, and TX for 2200-2020.
    *** For downloading irrigated fraction data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM use
    download_Irr_frac_from_IrrMapper_yearly() function.

    ########################
    # READ ME (for Irrigation Data)
    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020,
    whereas LANID data consists datasets of these 11 states and ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017.
    Our target year is 2000-2020. As LANID data isn't available for all the years we are using IrrMapper data as
    irrigated cropland data for 2000-2020 for the available 11 states' data (also it's available in GEE already).
    For the other 06 states, we clipped LANID data and compiled them as a GEE asset for 2000-2020 (2018-2020 data
    considered the same as 2017 data to fill the missing values). In summary, for WA, OR, CA, ID, NV, UT, AZ, MT, WY,
    CO, and NM (11 states) the irrigated field data comes from IrrMapper and for ND, SD, OK, KS, NE, and TX the
    irrigated field data comes from LANID. Also, adding AIM-HPA data with LANID to pick up irrigated fields that are
    seen in LANID data.
    ########################
.
    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    lanid_asset, _, _, _, _, _, _, _ = get_gee_dict(data_name)
    lanid_data_band_dict = {2000: 'lanid_2000', 2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003',
                            2004: 'lanid_2004', 2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007',
                            2008: 'lanid_2008', 2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011',
                            2012: 'lanid_2012', 2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015',
                            2016: 'lanid_2016', 2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019',
                            2020: 'lanid_2020'}

    # Extracting AIM-HPA dataset information (saved as an asset) from GEE
    AimHpa_asset, _, _, _, _, _, _, _ = get_gee_dict('AIM-HPA')
    aim_hpa_data_band_dict = {2000: 'aim_2000', 2001: 'aim_2001', 2002: 'aim_2002', 2003: 'aim_2003',
                              2004: 'aim_2004', 2005: 'aim_2005', 2006: 'aim_2006', 2007: 'aim_2007',
                              2008: 'aim_2008', 2009: 'aim_2009', 2010: 'aim_2010', 2011: 'aim_2011',
                              2012: 'aim_2012', 2013: 'aim_2013', 2014: 'aim_2014', 2015: 'aim_2015',
                              2016: 'aim_2016', 2017: 'aim_2017', 2018: 'aim_2018', 2019: 'aim_2019',
                              2020: 'aim_2020'}

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years
        print('********')
        print(f'Getting data urls for year={year}.....')

        # ######### Merging irrigated field dataset from LANID and AIM-HPA
        # # LANID data for the year
        # In LANID dataset irrigated fields are assigned as 1
        lanid_band = lanid_data_band_dict[year]
        irr_lanid = ee.Image(lanid_asset).select(lanid_band)
        irr_lanid = irr_lanid.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        projection_lanid = irr_lanid.projection()
        projection2km_scale = irr_lanid.projection().atScale(2200)  # 2km projection taken for LANID

        # # AIM-HPA data for the year
        # In AIM-HPA dataset irrigated fields are assigned as 1
        hpa_band = aim_hpa_data_band_dict[year]
        irr_hpa = ee.Image(AimHpa_asset).select(hpa_band)
        irr_hpa = irr_hpa.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        # # Joining LANID and AIM-HPA
        # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
        # Remapping the irrigated 0 values to 1 and then applying a mask
        irr_total = irr_lanid.add(irr_hpa)
        irr_total = irr_total.eq(0).remap([0], [1])
        irr_total = irr_total.updateMask(irr_total)
        # #########

        # In LANID + AIM-HPA dataset irrigated fields are assigned as 1
        # 30m Irrigation pixel count in each 2km pixel
        irr_total = irr_total.setDefaultProjection(crs=projection_lanid)
        irr_pixel_count = irr_total.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
                             .reproject(crs=projection2km_scale)

        # In LANID + AIM-HPA dataset irrigated fields are assigned as 1
        # Unmasking() to keep the irrigated values to 1 and setting others as 0
        irr_mask_with_total = irr_total.unmask()

        # Total number of 30m pixels in each 2km pixel
        total_pixel_count = irr_mask_with_total.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
            .reproject(crs=projection2km_scale)

        # counting fraction of irrigated lands in a pixel
        irrig_frac = irr_pixel_count.divide(total_pixel_count)

        # second loop for grids
        data_url_list = []
        local_file_paths_list = []

        for i in range(len(grid_no)):  # third loop for grids
            # converting grid geometry info to a GEE extent
            grid_sr = grid_no[i]
            roi = grid_geometry[i].bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            try:
                data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                      'crs': 'EPSG:4269',  # NAD83
                                                      'scale': 2200,  # in meter
                                                      'region': gee_extent,
                                                      'format': 'GEO_TIFF'})

            except:
                data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                      'crs': 'EPSG:4269',  # NAD83
                                                      'scale': 2200,  # in meter
                                                      'region': gee_extent,
                                                      'format': 'GEO_TIFF'})

            key_word = data_name
            local_file_path = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')

            # Appending data url and local file path (to save data) to a central list
            data_url_list.append(data_url)
            local_file_paths_list.append(local_file_path)

            # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
            # there is enough data url gathered for download.
            if (len(data_url_list) == 120) | (
                    i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                # Combining url and file paths together to pass in multiprocessing
                urls_to_file_paths_compile = []
                for j, k in zip(data_url_list, local_file_paths_list):
                    urls_to_file_paths_compile.append([j, k])

                # Download data by multi-processing/multi-threading
                download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                       use_cpu=use_cpu_while_multidownloading)

                # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                data_url_list = []
                local_file_paths_list = []


def download_CropET_from_OpenET_IrrMapper_monthly(data_name, download_dir, year_list, month_range, grid_shape,
                                                  scale=2200, use_cpu_while_multidownloading=15):
    """
    Download cropET data (at monthly scale) from OpenET GEE by filtering ET data with irrigated field data from
    IrrMapper.
    **** Don't use this function to download crop ET data for ND, SD, OK, KS, NE, and TX.
    Use download_CropET_from_OpenET_LANID_monthly()) function instead.

    ########################
    # READ ME (for Irrigation Data)
    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020,
    whereas LANID data consists datasets of these 11 states and ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017.
    Our target year is 2000-2020. As LANID data isn't available for all the years we are using IrrMapper data as
    irrigated cropland data for 2000-2020 for the available 11 states' data (also it's available in GEE already).
    For the other 06 states, we clipped LANID data and compiled them as a GEE asset for 2000-2020 (2018-2020 data
    considered the same as 2017 data to fill the missing values). In summary, for WA, OR, CA, ID, NV, UT, AZ, MT, WY,
    CO, and NM (11 states) the irrigated field data comes from IrrMapper and for ND, SD, OK, KS, NE, and TX the
    irrigated field data comes from LANID.
    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrig_crop_OpenET_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting IrrMapper and OpenET dataset information required for downloading from GEE
    et_data, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
    _, _ = get_gee_dict(data_name)

    irr_data, irr_band, irr_multiply_scale, irr_reducer, _, _, _, _ = get_gee_dict('IrrMapper')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    for year in year_list:  # first loop for years
        # # IrrMapper data for the year
        # In IrrMapper dataset irrigated fields are assigned as 0
        # Converting the irrigated values to 1 and setting others as nan
        # The mask will be applied on OpenET data to obtain cropET
        irrmap = ee.ImageCollection(irr_data).filter(ee.Filter.calendarRange(year, year, 'year')). \
            select(irr_band).reduce(irr_reducer)
        projection2km_scale = irrmap.projection().atScale(2200)  # 2km projection taken for IrrMapper

        irrig_filter = irrmap.eq(0)
        irr_mask = irrmap.updateMask(irrig_filter).remap([0], [1])

        for month in month_list:  # second loop for months
            print('********')
            print(f'Getting data urls for year={year}, month={month}.....')
            start_date = ee.Date.fromYMD(year, month, 1)
            start_date_dt = datetime(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
                end_date_dt = datetime(year, month + 1, 1)

            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= et_month_start_range) & (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(et_data)
                # getting default projection of OpenET
                projection_openET = ee.Image(openET_imcol.first()).projection()

                # getting image for year-month range.
                # the projection is lost during this image conversion, reapplying that at the end
                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date). \
                    reduce(et_reducer).multiply(et_multiply_scale).toFloat(). \
                    setDefaultProjection(crs=projection_openET)

                # multiplying OpenET with Irrmap irrigated data. This will set non-irrigated pixels' ET value to zero
                cropET_from_OpenET = openET_img.multiply(irr_mask)

                # summing crop ET (from openET) from 30m to 2km scale
                cropET_from_OpenET = cropET_from_OpenET. \
                    reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000). \
                    reproject(crs=projection2km_scale)

                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to retry
                    # failed connections
                    try:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})
                    except:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})

                    local_file_path = os.path.join(download_dir,
                                                   f'{data_name}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for i, j in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([i, j])

                        # Download data by multi-procesing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

            else:
                print(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_CropET_from_OpenET_LANID_monthly(data_name, download_dir, year_list, month_range, grid_shape, scale=2200,
                                              use_cpu_while_multidownloading=15):
    """
    Download cropET data (at monthly scale) from OpenET GEE by filtering ET data with irrigated field data from
    LANID.
    **** Don't use this function to download crop ET data for states except ND, SD, OK, KS, NE, and TX.
    Use download_CropET_from_OpenET_IrrMapper_monthly() function instead.

    ########################
    # READ ME (for Irrigation Data)
    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020,
    whereas LANID data consists datasets of these 11 states and ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017.
    Our target year is 2000-2020. As LANID data isn't available for all the years we are using IrrMapper data as
    irrigated cropland data for 2000-2020 for the available 11 states' data (also it's available in GEE already).
    For the other 06 states, we clipped LANID data and compiled them as a GEE asset for 2000-2020 (2018-2020 data
    considered the same as 2017 data to fill the missing values). In summary, for WA, OR, CA, ID, NV, UT, AZ, MT, WY,
    CO, and NM (11 states) the irrigated field data comes from IrrMapper and for ND, SD, OK, KS, NE, and TX the
    irrigated field data comes from LANID. Also, adding AIM-HPA data with LANID to pick up irrigated fields that are
    seen in LANID data.
    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrig_crop_OpenET_LANID']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting OpenET dataset information required for downloading from GEE
    et_data, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
    _, _ = get_gee_dict(data_name)

    # Extracting LANID dataset information (saved as an asset) from GEE
    lanid_asset, _, _, _, _, _, _, _ = get_gee_dict('LANID')
    lanid_data_band_dict = {2000: 'lanid_2000', 2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003',
                            2004: 'lanid_2004', 2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007',
                            2008: 'lanid_2008', 2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011',
                            2012: 'lanid_2012', 2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015',
                            2016: 'lanid_2016', 2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019',
                            2020: 'lanid_2020'}

    # Extracting AIM-HPA dataset information (saved as an asset) from GEE
    AimHpa_asset, _, _, _, _, _, _, _ = get_gee_dict('AIM-HPA')
    aim_hpa_data_band_dict = {2000: 'aim_2000', 2001: 'aim_2001', 2002: 'aim_2002', 2003: 'aim_2003',
                              2004: 'aim_2004', 2005: 'aim_2005', 2006: 'aim_2006', 2007: 'aim_2007',
                              2008: 'aim_2008', 2009: 'aim_2009', 2010: 'aim_2010', 2011: 'aim_2011',
                              2012: 'aim_2012', 2013: 'aim_2013', 2014: 'aim_2014', 2015: 'aim_2015',
                              2016: 'aim_2016', 2017: 'aim_2017', 2018: 'aim_2018', 2019: 'aim_2019',
                              2020: 'aim_2020'}

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    for year in year_list:  # first loop for years
        # ######### Merging irrigated field dataset from LANID and AIM-HPA
        # # LANID data for the year
        # In LANID dataset irrigated fields are assigned as 1
        lanid_band = lanid_data_band_dict[year]
        irr_lanid = ee.Image(lanid_asset).select(lanid_band)
        irr_lanid = irr_lanid.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        projection2km_scale = irr_lanid.projection().atScale(2200)  # 2km projection taken for LANID

        # # AIM-HPA data for the year
        # In AIM-HPA dataset irrigated fields are assigned as 1
        hpa_band = aim_hpa_data_band_dict[year]
        irr_hpa = ee.Image(AimHpa_asset).select(hpa_band)
        irr_hpa = irr_hpa.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        # # Joining LANID and AIM-HPA
        # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
        # Remapping the irrigated 0 values to 1 and then applying a mask
        irr_total = irr_lanid.add(irr_hpa)
        irr_total = irr_total.eq(0).remap([0], [1])
        irr_total = irr_total.updateMask(irr_total)
        # #########

        # second loop for months
        for month in month_list:
            print('********')
            print(f'Getting data urls for year={year}, month={month}.....')
            start_date = ee.Date.fromYMD(year, month, 1)
            start_date_dt = datetime(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
                end_date_dt = datetime(year, month + 1, 1)

            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= et_month_start_range) & (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(et_data)

                # getting default projection of OpenET
                projection_openET = ee.Image(openET_imcol.first()).projection()

                # getting image for year-month range.
                # the projection is lost during this image conversion, reapplying that at the end
                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date). \
                    reduce(et_reducer).multiply(et_multiply_scale).toFloat(). \
                    setDefaultProjection(crs=projection_openET)

                # multiplying OpenET with LANID _ AIm-HPA irrigated data. This will set non-irrigated pixels' ET value
                # to zero
                cropET_from_OpenET = openET_img.multiply(irr_total)

                # summing crop ET (from openET) from 30m to 2km scale
                cropET_from_OpenET = cropET_from_OpenET. \
                    reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000). \
                    reproject(crs=projection2km_scale)

                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to retry
                    # failed connections
                    try:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})
                    except:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})

                    local_file_path = os.path.join(download_dir,
                                                   f'{data_name}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for i, j in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([i, j])

                        # Download data by multi-procesing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

            else:
                print(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_Rainfed_frac_from_IrrMapper_yearly(data_name, download_dir, year_list, grid_shape,
                                                scale=2200, use_cpu_while_multidownloading=15):
    """
    Download IrrMapper-extent rainfed fraction data (at 2km scale) at yearly scale. The rainfed field data is created
    using CDL cropland and IrrMapper data.
    **** Don't use this function to download rainfed fraction data for ND, SD, OK, KS, NE, and TX.
    Use download_Rainfed_frac_from_LANID_yearly() function instead.

    ########################
    # READ ME (for Rainfed Data)
    The rainfed fields/pixels dataset has been created combining CDL cropland data with IrrMapper and LANID datasets.
    IrrMapper data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020, whereas,
    LANID data consists datasets of these 11 states and ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017
    (extrapolated for 2018-2020 considering them the same as 2017 data). The CDL data for entire CONUS is available from
     2008. Therefore, we have to create rainfed field/pixel data in multiple steps-
        1. Combine CDL cropland and IrrMapper to get rainfed data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM
            (11 states) for 2008-2020
        2. Combine CDL cropland and LANID to get rainfed data for ND, SD, OK, KS, NE, and TX (06 states) for 2008-2020
        3. Download rainfed field data (or fraction of rainfed field in each 2km pixel) for 2008-2020 from GEE and do
            binary classification (1- rainfed, 0- irrigated/some other type)
        4. Extend rainfed data for 2000-2007 by maximum occurrence approach.
    ########################
    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Rainfed_Frac_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for. Should be within 2016 to 2020.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting IrrMapper and CDL dataset information required for downloading from GEE
    irr_data, irr_band, irr_multiply_scale, irr_reducer, _, _, _, _ = get_gee_dict('IrrMapper')
    cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, \
    cdl_year_start_date, cdl_year_end_date = get_gee_dict('USDA_CDL')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    for year in year_list:  # first loop for years
        print('********')
        print(f'Getting data urls for year={year}.....')

        # # # # # # # # # Creating Rainfed dataset by combining IrrMapper and USDA CDL data

        # # IrrMapper data for the year
        irrmap_imcol = ee.ImageCollection(irr_data)
        irrmapper_img = irrmap_imcol.filter(ee.Filter.calendarRange(year, year, 'year')).select(irr_band) \
            .reduce(irr_reducer)
        # In IrrMapper dataset irrigated fields are assigned as 0, converting it to 1 and others to 0
        irrmapper_img = irrmapper_img.eq(0)

        # Extracting IrrMapper projection
        projection_irrmap = irrmap_imcol.first().projection()
        projection2km_scale = projection_irrmap.atScale(2200)  # 2km projection taken for IrrMapper

        # # USDA CDL data for the year
        cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first().select(cdl_band)
        #  List of non-crop pixels
        noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88,
                                111, 112, 121, 122, 123,
                                124, 131, 141, 142, 143,
                                152, 190, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class
        # Filtering out non-crop pixels
        cdl_cropland = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

        # Extracting rainfed (non-irrigated) croplands
        # Converting 0 values (non-irrigated) to 1 then multiplying with
        # CDL cropland data (reclassified to 1 for croplands and 0 for non-croplands).
        # The remaining pixels will be rainfed croplands. Masking it again to remove the 0 values
        irrmapper_reversed = irrmapper_img.remap([0], [1])
        rainfed_cropland_for_IrrMapper = cdl_cropland.multiply(irrmapper_reversed)
        mask = rainfed_cropland_for_IrrMapper.eq(1)
        rainfed_cropland_for_IrrMapper = rainfed_cropland_for_IrrMapper.updateMask(mask) \
            .setDefaultProjection(crs=projection_irrmap)

        # Counting number of rainfed pixel in each 2km pixel
        rainfed_pixel_count = rainfed_cropland_for_IrrMapper.reduceResolution(reducer=ee.Reducer.count(),
                                                                              maxPixels=60000) \
            .reproject(crs=projection2km_scale)

        # Unmaking the rainfed pixels so that non-rainfed pixels have 0 value
        rainfed_with_total = rainfed_cropland_for_IrrMapper.unmask()

        # total 30m pixel count in 2km pixel
        total_pixel_count = rainfed_with_total.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
            .reproject(crs=projection2km_scale)

        # counting fraction of irrigated lands in a pixel
        rainfed_frac_IrrMapper = rainfed_pixel_count.divide(total_pixel_count)
        # # # # # # # # #

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        start_date_dt = datetime(year, 1, 1)
        end_date_dt = datetime(year + 1, 1, 1)

        if (start_date_dt >= cdl_year_start_date) & (end_date_dt <= cdl_year_end_date):
            # will collect url and file name in url list and local_file_paths_list
            data_url_list = []
            local_file_paths_list = []

            for i in range(len(grid_no)):  # second loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Getting Data URl for each grid from GEE
                # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to retry
                # failed connections
                try:
                    data_url = rainfed_frac_IrrMapper.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})
                except:
                    data_url = rainfed_frac_IrrMapper.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})

                keyword = data_name
                local_file_path = os.path.join(download_dir, f'{keyword}_{str(year)}_{str(grid_sr)}.tif')

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for i, j in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([i, j])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

        else:
            print(f'Data for year {year} is out of range. Skipping query')
            pass


def download_Rainfed_frac_from_LANID_yearly(data_name, download_dir, year_list, grid_shape,
                                            scale=2200, use_cpu_while_multidownloading=15):
    """
    Download IrrMapper-extent rainfed fraction data (at 2km scale) at yearly scale. The rainfed field data is created
    using CDL cropland and IrrMapper data.
    **** Don't use this function to download rainfed fraction data for ND, SD, OK, KS, NE, and TX.
    Use download_Rainfed_frac_from_LANID_yearly() function instead.

    ########################
    # READ ME (for Rainfed Data)
    The rainfed fields/pixels dataset has been created combining CDL cropland data with IrrMapper and LANID + AIM-HPA
    datasets. IrrMapper data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020,
    whereas, LANID data consists datasets of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017 (extrapolated for
    2018-2020 considering them the same as 2017 data). The CDL data for entire CONUS is available from 2008. Therefore,
    we have to create rainfed field/pixel data in multiple steps.
        1. Combine CDL cropland and IrrMapper to get rainfed data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM
            (11 states) for 2008-2020
        2. Combine LANID + AIM-HPA data and compare with CDL cropland rainfed data for ND, SD, OK, KS, NE, and TX
            (06 states) for 2008-2020
        3. Download rainfed field data (or fraction of rainfed field in each 2km pixel) for 2008-2020 from GEE and do
            binary classification (1- rainfed, 0- irrigated/some other type)
        4. Extend rainfed data for 2000-2007 by maximum occurrence approach.
    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Rainfed_Frac_LANID']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for. Should be within 2016 to 2020.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting LANID dataset information (saved as an asset) from GEE
    lanid_asset, _, _, _, _, _, _, _ = get_gee_dict('LANID')
    lanid_data_band_dict = {2000: 'lanid_2000', 2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003',
                            2004: 'lanid_2004', 2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007',
                            2008: 'lanid_2008', 2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011',
                            2012: 'lanid_2012', 2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015',
                            2016: 'lanid_2016', 2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019',
                            2020: 'lanid_2020'}

    # Extracting AIM-HPA dataset information (saved as an asset) from GEE
    AimHpa_asset, _, _, _, _, _, _, _ = get_gee_dict('AIM-HPA')
    aim_hpa_data_band_dict = {2000: 'aim_2000', 2001: 'aim_2001', 2002: 'aim_2002', 2003: 'aim_2003',
                              2004: 'aim_2004', 2005: 'aim_2005', 2006: 'aim_2006', 2007: 'aim_2007',
                              2008: 'aim_2008', 2009: 'aim_2009', 2010: 'aim_2010', 2011: 'aim_2011',
                              2012: 'aim_2012', 2013: 'aim_2013', 2014: 'aim_2014', 2015: 'aim_2015',
                              2016: 'aim_2016', 2017: 'aim_2017', 2018: 'aim_2018', 2019: 'aim_2019',
                              2020: 'aim_2020'}

    # Extracting CDL dataset information required for downloading from GEE
    cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, \
    cdl_year_start_date, cdl_year_end_date = get_gee_dict('USDA_CDL')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    for year in year_list:  # first loop for years
        print('********')
        print(f'Getting data urls for year={year}.....')

        # ######### Merging irrigated field dataset from LANID and AIM-HPA
        # # LANID data for the year
        # In LANID dataset irrigated fields are assigned as 1
        lanid_band = lanid_data_band_dict[year]
        irr_lanid = ee.Image(lanid_asset).select(lanid_band)
        irr_lanid = irr_lanid.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        projection_lanid = irr_lanid.projection()
        projection2km_scale = irr_lanid.projection().atScale(2200)  # 2km projection taken for LANID

        # # AIM-HPA data for the year
        # In AIM-HPA dataset irrigated fields are assigned as 1
        hpa_band = aim_hpa_data_band_dict[year]
        irr_hpa = ee.Image(AimHpa_asset).select(hpa_band)
        irr_hpa = irr_hpa.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        # # Joining LANID and AIM-HPA
        # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
        # Remapping the irrigated 0 values to 1 and then applying a mask
        irr_total = irr_lanid.add(irr_hpa)
        irr_total = irr_total.eq(0).remap([0], [1]).unmask()
        # #########

        # # USDA CDL data for the year
        cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first().select(cdl_band)
        #  List of non-crop pixels
        noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88,
                                111, 112, 121, 122, 123, 124,
                                131, 141, 142, 143, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class
        # Filtering out non-crop pixels
        cdl_cropland = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

        # Creating Rainfed dataset by combining LANID + AIM-HPA and USDA CDL data
        # Extracting rainfed (non-irrigated) croplands
        # Converting 0 values (non-irrigated) to 1, and 1 (irrigated) to 0
        # Then multiplying with CDL cropland data (reclassified to 1 for croplands and 0 for non-croplands).
        # The remaining pixels will be rainfed croplands. Masking it again to remove the 0 values
        irr_total_reversed = irr_total.remap([0, 1], [1, 0])
        rainfed_cropland = cdl_cropland.multiply(irr_total_reversed)
        mask = rainfed_cropland.eq(1)
        rainfed_cropland = rainfed_cropland.updateMask(mask).setDefaultProjection(crs=projection_lanid)

        # Counting number of rainfed pixel in each 2km pixel
        rainfed_pixel_count = rainfed_cropland.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
                                                        .reproject(crs=projection2km_scale)

        # Unmaking the rainfed pixels so that non-rainfed pixels have 0 value
        rainfed_with_total = rainfed_cropland.unmask()

        # total 30m pixel count in 2km pixel
        total_pixel_count = rainfed_with_total.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
            .reproject(crs=projection2km_scale)

        # counting fraction of irrigated lands in a pixel
        rainfed_frac_LANID = rainfed_pixel_count.divide(total_pixel_count)
        # # # # # # # # #

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        start_date_dt = datetime(year, 1, 1)
        end_date_dt = datetime(year + 1, 1, 1)

        if (start_date_dt >= cdl_year_start_date) & (end_date_dt <= cdl_year_end_date):
            # will collect url and file name in url list and local_file_paths_list
            data_url_list = []
            local_file_paths_list = []

            for i in range(len(grid_no)):  # second loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Getting Data URl for each grid from GEE
                # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to retry
                # failed connections
                try:
                    data_url = rainfed_frac_LANID.getDownloadURL({'name': data_name,
                                                                  'crs': 'EPSG:4269',  # NAD83
                                                                  'scale': scale,  # in meter. equal to ~0.02 deg
                                                                  'region': gee_extent,
                                                                  'format': 'GEO_TIFF'})
                except:
                    data_url = rainfed_frac_LANID.getDownloadURL({'name': data_name,
                                                                  'crs': 'EPSG:4269',  # NAD83
                                                                  'scale': scale,  # in meter. equal to ~0.02 deg
                                                                  'region': gee_extent,
                                                                  'format': 'GEO_TIFF'})

                keyword = data_name
                local_file_path = os.path.join(download_dir, f'{keyword}_{str(year)}_{str(grid_sr)}.tif')

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for i, j in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([i, j])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

        else:
            print(f'Data for year {year} is out of range. Skipping query')
            pass


def download_Rainfed_CropET_from_OpenET_IrrMapper_monthly(data_name, download_dir, year_list, month_range, grid_shape,
                                                          scale=2200, use_cpu_while_multidownloading=15):
    """
    Download cropET data (at monthly scale) from OpenET GEE by filtering ET data with rainfed field data. The rainfed
    field data is created using CDL cropland and IrrMapper data.
    **** Don't use this function to download rainfed cropET data for ND, SD, OK, KS, NE, and TX.
    Use download_Rainfed_CropET_from_OpenET_LANID_monthly() function instead.

    ########################
    # READ ME (for Rainfed Data)
    The rainfed fields/pixels dataset has been created combining CDL cropland data with IrrMapper and LANID datasets.
    IrrMapper data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020, whereas,
    LANID data consists datasets of these 11 states and ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017
    (extrapolated for 2018-2020 considering them the same as 2017 data). The CDL data for entire CONUS is available from
     2008. Therefore, we have to create rainfed field/pixel data in multiple steps-
        1. Combine CDL cropland and IrrMapper to get rainfed data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM
            (11 states) for 2008-2020
        2. Combine CDL cropland and LANID to get rainfed data for ND, SD, OK, KS, NE, and TX (06 states) for 2008-2020
        3. Download rainfed field data (or fraction of rainfed field in each 2km pixel) for 2008-2020 from GEE and do
            binary classification (1- rainfed, 0- irrigated/some other type)
        4. Extend rainfed data for 2000-2007 by maximum occurrence approach.
    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Rainfed_crop_OpenET_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting OpenET, CDL, IrrMapper dataset information required for downloading from GEE
    et_data, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
    _, _ = get_gee_dict(data_name)

    irr_data, irr_band, irr_multiply_scale, irr_reducer, _, _, _, _ = get_gee_dict('IrrMapper')
    cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, _, _ = get_gee_dict('USDA_CDL')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    for year in year_list:  # first loop for years
        # # # # # # # # # Creating Rainfed dataset by combining IrrMapper and USDA CDL data

        # # IrrMapper data for the year
        irrmap_imcol = ee.ImageCollection(irr_data)
        irrmapper_img = irrmap_imcol.filter(ee.Filter.calendarRange(year, year, 'year')).select(irr_band) \
            .reduce(irr_reducer)
        # In IrrMapper dataset irrigated fields are assigned as 0, converting it to 1 and others to nan
        irrmapper_img = irrmapper_img.eq(0)

        # Extracting IrrMapper projection
        projection_irrmap = irrmapper_img.projection()
        projection2km_scale = projection_irrmap.atScale(2200)  # 2km projection taken for IrrMapper

        # # USDA CDL data for the year
        cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first().select(cdl_band)
        #  List of non-crop pixels
        noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111,
                                112, 121, 122, 123, 124, 131, 141, 142,
                                143, 152, 190, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class
        # Filtering out non-crop pixels
        cdl_cropland = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

        # Extracting rainfed (non-irrigated) croplands
        # Converting 0 values (non-irrigated) to 1 then multiplying with
        # CDL cropland data (reclassified to 1 for croplands and 0 for non-croplands).
        # The remaining pixels will be rainfed croplands. Masking it again to remove the 0 values
        irrmapper_reversed = irrmapper_img.remap([0], [1])
        rainfed_cropland_for_IrrMapper = cdl_cropland.multiply(irrmapper_reversed)
        mask = rainfed_cropland_for_IrrMapper.eq(1)
        rainfed_cropland_for_IrrMapper = rainfed_cropland_for_IrrMapper.updateMask(mask) \
            .setDefaultProjection(crs=projection_irrmap)
        # # # # # # # # #

        for month in month_list:  # second loop for months
            print('********')
            print(f'Getting data urls for year={year}, month={month}.....')
            start_date = ee.Date.fromYMD(year, month, 1)
            start_date_dt = datetime(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
                end_date_dt = datetime(year, month + 1, 1)

            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= et_month_start_range) & (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(et_data)
                # getting default projection of OpenET
                projection_openET = ee.Image(openET_imcol.first()).projection()

                # getting image for year-month range.
                # the projection is lost during this image conversion, reapplying that at the end
                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date). \
                    reduce(et_reducer).multiply(et_multiply_scale).toFloat(). \
                    setDefaultProjection(crs=projection_openET)

                # multiplying OpenET with IrrMapper-CDL Rainfed data. This will set irrigated pixels' ET value to zero
                cropET_from_OpenET = openET_img.multiply(rainfed_cropland_for_IrrMapper)

                # summing crop ET (from openET) from 30m to 2km scale
                cropET_from_OpenET = cropET_from_OpenET. \
                    reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000). \
                    reproject(crs=projection2km_scale)

                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to retry
                    # failed connections
                    try:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})
                    except:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})

                    local_file_path = os.path.join(download_dir,
                                                   f'{data_name}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for i, j in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([i, j])

                        # Download data by multi-processing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

            else:
                print(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_Rainfed_CropET_from_OpenET_LANID_monthly(data_name, download_dir, year_list, month_range, grid_shape,
                                                      scale=2200, use_cpu_while_multidownloading=15):
    """
    Download cropET data (at monthly scale) from OpenET GEE by filtering ET data with rainfed field data. The rainfed
    field data is created using CDL cropland and LANID data.
    **** Don't use this function to download rainfed cropET data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM.
    Use download_Rainfed_CropET_from_OpenET_IrrMapper_monthly() function instead.

    ########################
    # READ ME (for Rainfed Data)
    The rainfed fields/pixels dataset has been created combining CDL cropland data with IrrMapper and LANID + AIM-HPA
    datasets. IrrMapper data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 2000-2020,
    whereas, LANID data consists datasets of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2017 (extrapolated for
    2018-2020 considering them the same as 2017 data). The CDL data for entire CONUS is available from 2008. Therefore,
    we have to create rainfed field/pixel data in multiple steps.
        1. Combine CDL cropland and IrrMapper to get rainfed data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM
            (11 states) for 2008-2020
        2. Combine LANID + AIM-HPA data and compare with CDL cropland rainfed data for ND, SD, OK, KS, NE, and TX
            (06 states) for 2008-2020
        3. Download rainfed field data (or fraction of rainfed field in each 2km pixel) for 2008-2020 from GEE and do
            binary classification (1- rainfed, 0- irrigated/some other type)
        4. Extend rainfed data for 2000-2007 by maximum occurrence approach.
    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Rainfed_crop_OpenET_LANID']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting LANID dataset information (saved as an asset) from GEE
    lanid_asset, _, _, _, _, _, _, _ = get_gee_dict('LANID')
    lanid_data_band_dict = {2000: 'lanid_2000', 2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003',
                            2004: 'lanid_2004', 2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007',
                            2008: 'lanid_2008', 2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011',
                            2012: 'lanid_2012', 2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015',
                            2016: 'lanid_2016', 2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019',
                            2020: 'lanid_2020'}

    # Extracting AIM-HPA dataset information (saved as an asset) from GEE
    AimHpa_asset, _, _, _, _, _, _, _ = get_gee_dict('AIM-HPA')
    aim_hpa_data_band_dict = {2000: 'aim_2000', 2001: 'aim_2001', 2002: 'aim_2002', 2003: 'aim_2003',
                              2004: 'aim_2004', 2005: 'aim_2005', 2006: 'aim_2006', 2007: 'aim_2007',
                              2008: 'aim_2008', 2009: 'aim_2009', 2010: 'aim_2010', 2011: 'aim_2011',
                              2012: 'aim_2012', 2013: 'aim_2013', 2014: 'aim_2014', 2015: 'aim_2015',
                              2016: 'aim_2016', 2017: 'aim_2017', 2018: 'aim_2018', 2019: 'aim_2019',
                              2020: 'aim_2020'}

    # Extracting OpenET and CDL dataset information required for downloading from GEE
    et_data, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
    _, _ = get_gee_dict(data_name)

    cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, _, _ = get_gee_dict('USDA_CDL')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    for year in year_list:  # first loop for years
        # ######### Merging irrigated field dataset from LANID and AIM-HPA
        # # LANID data for the year
        # In LANID dataset irrigated fields are assigned as 1
        lanid_band = lanid_data_band_dict[year]
        irr_lanid = ee.Image(lanid_asset).select(lanid_band)
        irr_lanid = irr_lanid.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        projection_lanid = irr_lanid.projection()
        projection2km_scale = irr_lanid.projection().atScale(2200)  # 2km projection taken for LANID

        # # AIM-HPA data for the year
        # In AIM-HPA dataset irrigated fields are assigned as 1
        hpa_band = aim_hpa_data_band_dict[year]
        irr_hpa = ee.Image(AimHpa_asset).select(hpa_band)
        irr_hpa = irr_hpa.eq(1).unmask()  # unmask() assigns zero value to no data pixels

        # # Joining LANID and AIM-HPA
        # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
        # Remapping the irrigated 0 values to 1 and then applying a mask
        irr_total = irr_lanid.add(irr_hpa)
        irr_total = irr_total.eq(0).remap([0], [1]).unmask()
        # #########

        # # USDA CDL data for the year
        cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first().select(cdl_band)
        #  List of non-crop pixels
        noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88,
                                111, 112, 121, 122, 123, 124,
                                131, 141, 142, 143, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class
        # Filtering out non-crop pixels
        cdl_cropland = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

        # Creating Rainfed dataset by combining LANID + AIM-HPA and USDA CDL data
        # Extracting rainfed (non-irrigated) croplands
        # Converting 0 values (non-irrigated) to 1, and 1 (irrigated) to 0
        # Then multiplying with CDL cropland data (reclassified to 1 for croplands and 0 for non-croplands).
        # The remaining pixels will be rainfed croplands. Masking it again to remove the 0 values
        irr_total_reversed = irr_total.remap([0, 1], [1, 0])
        rainfed_cropland = cdl_cropland.multiply(irr_total_reversed)
        mask = rainfed_cropland.eq(1)
        rainfed_cropland = rainfed_cropland.updateMask(mask).setDefaultProjection(crs=projection_lanid)

        for month in month_list:  # second loop for months
            print('********')
            print(f'Getting data urls for year={year}, month={month}.....')
            start_date = ee.Date.fromYMD(year, month, 1)
            start_date_dt = datetime(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
                end_date_dt = datetime(year, month + 1, 1)

            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= et_month_start_range) & (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(et_data)
                # getting default projection of OpenET
                projection_openET = ee.Image(openET_imcol.first()).projection()

                # getting image for year-month range.
                # the projection is lost during this image conversion, reapplying that at the end
                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date). \
                    reduce(et_reducer).multiply(et_multiply_scale).toFloat(). \
                    setDefaultProjection(crs=projection_openET)

                # multiplying OpenET with LANID Rainfed data. This will set irrigated pixels' ET value to zero
                cropET_from_OpenET = openET_img.multiply(rainfed_cropland)

                # summing crop ET (from openET) from 30m to 2km scale
                cropET_from_OpenET = cropET_from_OpenET. \
                    reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000). \
                    reproject(crs=projection2km_scale)

                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to retry
                    # failed connections
                    try:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})
                    except:
                        data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                      'crs': 'EPSG:4269',  # NAD83
                                                                      'scale': scale,  # in meter. equal to ~0.02 deg
                                                                      'region': gee_extent,
                                                                      'format': 'GEO_TIFF'})

                    local_file_path = os.path.join(download_dir,
                                                   f'{data_name}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for i, j in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([i, j])

                        # Download data by multi-procesing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

            else:
                print(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_eff_precip_data_from_DK_asset(data_name, download_dir, year_list, month_range, grid_shape,
                                           merge_keyword='WestUS_monthly', use_cpu_while_multidownloading=2,
                                           refraster=WestUS_raster):
    """
    Download Effective precipitation estimate by David Ketchum.
.
    :param data_name: Data name which will be used to create download directoryand name the data. Set to
                      'Effect_precip_DK'.
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param merge_keyword: Keyword to use for merging downloaded data. Default to 'WestUS_monthly'.
    :param refraster: Reference raster to use for merging downloaded datasets.

    :return: None.
    """
    ee.Initialize()
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years
        for month in month_list:  # second loop for months
            print('********')
            print(f'Getting data urls for year={year}, month={month}.....')

            # the band name changes with year and month
            # selecting the right data and band
            data_band = f'users/dgketchum/expansion/ept/ept_image_full_stack_{year}_et_{month}'

            # querying the data. original unit in m/month. multiplying with 100 to convert to mm/month
            download_data = ee.Image(data_band).multiply(1000).toFloat()

            # will collect url and file name in url list and local_file_paths_list
            data_url_list = []
            local_file_paths_list = []

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Getting Data URl for each grid from GEE
                # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to
                # retry failed connections
                try:
                    data_url = download_data.getDownloadURL({'name': data_name,
                                                             'crs': 'EPSG:4269',  # NAD83
                                                             'scale': 2200,  # in meter. equal to ~0.02 deg
                                                             'region': gee_extent,
                                                             'format': 'GEO_TIFF'})
                except:
                    data_url = download_data.getDownloadURL({'name': data_name,
                                                             'crs': 'EPSG:4269',  # NAD83
                                                             'scale': 2200,  # in meter. equal to ~0.02 deg
                                                             'region': gee_extent,
                                                             'format': 'GEO_TIFF'})

                key_word = data_name
                local_file_path = os.path.join(download_dir,
                                               f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for j, k in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([j, k])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

            mosaic_name = f'{data_name}_{year}_{month}.tif'
            mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
            makedirs([mosaic_dir])
            search_by = f'*{year}_{month}*.tif'
            mosaic_rasters_from_directory(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster,
                                          search_by=search_by, nodata=no_data_value)
            print(f'{data_name} monthly data downloaded and merged')

        else:
            print(f'Data for year {year}, month {month} is out of range. Skipping query')
            pass


def download_all_gee_data(data_list, download_dir, year_list, month_range,
                          grid_shape_large, grid_shape_for30m_irrmapper, grid_shape_for30m_lanid,
                          use_cpu_while_multidownloading=15, skip_download=False):
    """
    Used to download all gee data together.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['GRIDMET_Precip', 'MODIS_Day_LST', 'MODIS_NDWI', 'MODIS_NDVI', 'MODIS_LAI',
        'TERRACLIMATE_RET', 'TERRACLIMATE_vap_pres', 'TERRACLIMATE_vap_pres_def',
        'GRIDMET_max_RH', 'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad',
        'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'DAYMET_sun_hr',
        'Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID', 'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID',
        'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID', 'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID',
        'USDA_CDL', 'Field_capacity', 'Bulk_density', 'Organic_carbon_content', 'Sand_content', 'Clay_content', 'DEM',
        'Effect_precip_DK']
        ******************************

    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_large: File path of larger grids (used in most GEE data download) for which data
                             will be downloaded and mosaiced.
    :param grid_shape_for30m_irrmapper: File path of smaller grids to download data for IrrMapper extent and cropET from
                                        openET (this datasets are processed at 30m res in GEE, so smaller grids are
                                        required).
    :param grid_shape_for30m_lanid: File path of smaller grids to download data for LANID (for 6 central states) extent
                                    and cropET from openET (this datasets are processed at 30m res in GEE, so smaller
                                    grids are required).
    :param refraster_westUS: Reference raster to use for merging downloaded datasets.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param skip_download: Set to True to skip download.

    :return: None
    """
    if not skip_download:
        for data_name in data_list:

            if data_name not in ['Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID',
                                 'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID',
                                 'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID',
                                 'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID', 'USDA_CDL',
                                 'Field_capacity', 'Bulk_density', 'Organic_carbon_content',
                                 'Sand_content', 'Clay_content', 'DEM', 'Effect_precip_DK', 'Tree_cover']:
                # for datasets that needed to be downloaded on monthly scale
                download_gee_data_monthly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                          month_range=month_range, merge_keyword='WestUS_monthly',
                                          refraster_westUS=WestUS_raster,
                                          refraster_gee_merge=GEE_merging_refraster_large_grids,
                                          grid_shape=grid_shape_large,
                                          use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'USDA_CDL':
                download_gee_data_yearly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                         month_range=month_range, merge_keyword='WestUS_yearly',
                                         grid_shape=grid_shape_large, refraster_westUS=WestUS_raster,
                                         refraster_gee_merge=GEE_merging_refraster_large_grids)

            elif data_name == 'Irrig_crop_OpenET_IrrMapper':
                download_CropET_from_OpenET_IrrMapper_monthly(data_name=data_name, download_dir=download_dir,
                                                              year_list=year_list, month_range=month_range,
                                                              grid_shape=grid_shape_for30m_irrmapper, scale=2200,
                                                              use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrig_crop_OpenET_LANID':
                download_CropET_from_OpenET_LANID_monthly(data_name=data_name, download_dir=download_dir,
                                                          year_list=year_list, month_range=month_range,
                                                          grid_shape=grid_shape_for30m_lanid, scale=2200,
                                                          use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrigation_Frac_IrrMapper':
                download_Irr_frac_from_IrrMapper_yearly(data_name=data_name, download_dir=download_dir,
                                                        year_list=year_list, grid_shape=grid_shape_for30m_irrmapper,
                                                        use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrigation_Frac_LANID':
                download_Irr_frac_from_LANID_yearly(data_name=data_name, download_dir=download_dir,
                                                    year_list=year_list, grid_shape=grid_shape_for30m_lanid,
                                                    use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Rainfed_crop_OpenET_IrrMapper':
                download_Rainfed_CropET_from_OpenET_IrrMapper_monthly(data_name=data_name, download_dir=download_dir,
                                                                      year_list=year_list, month_range=month_range,
                                                                      grid_shape=grid_shape_for30m_irrmapper,
                                                                      use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Rainfed_crop_OpenET_LANID':
                download_Rainfed_CropET_from_OpenET_LANID_monthly(data_name=data_name, download_dir=download_dir,
                                                                  year_list=year_list, month_range=month_range,
                                                                  grid_shape=grid_shape_for30m_lanid,
                                                                  use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Rainfed_Frac_IrrMapper':
                download_Rainfed_frac_from_IrrMapper_yearly(data_name=data_name, download_dir=download_dir,
                                                            year_list=year_list, grid_shape=grid_shape_for30m_irrmapper,
                                                            use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Rainfed_Frac_LANID':
                download_Rainfed_frac_from_LANID_yearly(data_name=data_name, download_dir=download_dir,
                                                        year_list=year_list, grid_shape=grid_shape_for30m_lanid,
                                                        use_cpu_while_multidownloading=use_cpu_while_multidownloading)
            elif data_name in ['Field_capacity', 'Bulk_density', 'Organic_carbon_content', 'Sand_content',
                               'Clay_content']:
                download_soil_datasets(data_name=data_name, download_dir=download_dir, merge_keyword='WestUS',
                                       grid_shape=grid_shape_large, refraster_westUS=WestUS_raster)

            elif data_name == 'DEM':
                download_DEM_Slope_data(data_name=data_name, download_dir=download_dir,
                                        merge_keyword='WestUS', grid_shape=grid_shape_large,
                                        refraster_westUS=WestUS_raster,
                                        refraster_gee_merge=GEE_merging_refraster_large_grids,
                                        terrain_slope=False)

            elif data_name == 'Tree_cover':
                download_tree_cover_data(data_name='Tree_cover', download_dir=download_dir,
                                         merge_keyword='WestUS', grid_shape=grid_shape_large,
                                         refraster_westUS=WestUS_raster,
                                         refraster_gee_merge=GEE_merging_refraster_large_grids,
                                         westUS_shape=WestUS_shape)

            elif data_name == 'Effect_precip_DK':
                download_eff_precip_data_from_DK_asset(data_name=data_name, download_dir=download_dir,
                                                       year_list=year_list, month_range=(4, 10),
                                                       grid_shape=grid_shape_large, merge_keyword='WestUS_monthly',
                                                       use_cpu_while_multidownloading=use_cpu_while_multidownloading,
                                                       refraster=WestUS_raster)

    else:
        pass


def download_ssebop_et(years_list, month_range_list, download_dir='../../Data_main/Raster_data/Ssebop_ETa',
                       ssebop_link='https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads/',
                       skip_download=False):
    """
    Download ssebop actual ET data (unit in mm).

    :param years_list: List of years for which to download data.
    :param month_range_list: List of first and last month, i.e., to download data from April-September use [4, 9].
    :param download_dir: Directory path to download data.
    :param ssebop_link: USGS link with ssebop ET data.
    :param skip_download: Set to True to skip download.

    :return: None.
    """
    makedirs([download_dir])
    if not skip_download:
        months_list = [m for m in range(month_range_list[0], month_range_list[1] + 1)]

        year_month_combs = list(itertools.product(years_list, months_list))

        for each in year_month_combs:
            year, month = each
            print(f'Downloading SSEBOP ET for year {year} month {month}...')

            if len(str(month)) == 1:
                ssebop_et_link = f'{ssebop_link}m{str(year)}0{str(month)}.zip'
            else:
                ssebop_et_link = f'{ssebop_link}m{str(year)}{str(month)}.zip'
            r = requests.get(ssebop_et_link, allow_redirects=True)

            download_name = ssebop_et_link[ssebop_et_link.rfind('/') + 1:]
            download_to = os.path.join(download_dir, download_name)

            open(download_to, 'wb').write(r.content)

        zipped_files = extract_data(download_dir, download_dir, search_by='*.zip', rename_file=True)
        for z in zipped_files:
            os.remove(z)


def download_all_datasets(year_list, month_range,
                          grid_shape_large, grid_shape_for30m_irrmapper, grid_shape_for30m_lanid,
                          gee_data_list=None,
                          data_download_dir='../../Data_main/Raster_data',
                          skip_download_gee_data=True, use_cpu_while_multidownloading=15,
                          skip_download_ssebop_data=True):
    """
    Download all GEE datasets and ssebop data.

    :param year_list: List of years to download data for. We will use data for [2010, 2015] in the model.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_large: File path of larger grids (used in most GEE data download) for which data
                             will be downloaded and mosaiced.
    :param grid_shape_for30m_irrmapper: File path of smaller grids to download data for IrrMapper extent and cropET from
                                        openET (this datasets are processed at 30m res in GEE, so smaller grids are
                                        required).
    :param grid_shape_for30m_lanid: File path of smaller grids to download data for LANID (for 6 central states) extent
                                    and cropET from openET (this datasets are processed at 30m res in GEE, so smaller
                                    grids are required).
    :param gee_data_list: List of data to download from GEE. Default set to None, use if skip_download_gee_data=True.
                          Datasets currently downloaded by this code:
                          ['GRIDMET_Precip', 'OpenET_ensemble', 'MODIS_Day_LST', 'MODIS_NDWI', 'MODIS_LAI', 'MODIS_NDVI',
                           'TERRACLIMATE_vap_pres', 'GRIDMET_RET', 'GRIDMET_vap_pres_def',  'GRIDMET_max_RH',
                           'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                           'Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID', 'Irrigation_Frac_IrrMapper',
                           'Irrigation_Frac_LANID', 'Rainfed_crop_OpenET_IrrMapper', 'Rainfed_crop_OpenET_LANID',
                           'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID', 'USDA_CDL', 'Field_capacity',
                           'Bulk_density', 'Organic_carbon_content','Sand_content', 'Clay_content', 'DEM',
                           'Effect_precip_DK', 'Tree_cover']
    :param data_download_dir: Directory path to download and save data.
    :param skip_download_gee_data: Set to False if want to download listed data. Default set to True.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param skip_download_ssebop_data: Set to False if want to download ssebop data. Default set to True.

    :return: None.
    """
    # Data download from GEE
    download_all_gee_data(gee_data_list, download_dir=data_download_dir,
                          year_list=year_list, month_range=month_range,
                          grid_shape_large=grid_shape_large, grid_shape_for30m_irrmapper=grid_shape_for30m_irrmapper,
                          grid_shape_for30m_lanid=grid_shape_for30m_lanid, skip_download=skip_download_gee_data,
                          use_cpu_while_multidownloading=use_cpu_while_multidownloading)

    # SseBop data download from USGS link
    ssebop_download_dir = os.path.join(data_download_dir, 'Ssebop_ETa')
    download_ssebop_et(year_list, month_range_list=[1, 12],
                       download_dir=ssebop_download_dir,
                       skip_download=skip_download_ssebop_data)


