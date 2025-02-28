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
model_res = 2000  # in m
AZ_shape = '../../Data_main/AZ_files/ref_files/AZ.shp'
AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'
GEE_merging_refraster_large_grids = '../../Data_main/AZ_files/ref_files/AZ_gee_merge_ref_raster.tif'


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
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    gee_data_dict = {
        'SMAP_SM': 'NASA_USDA/HSL/SMAP10KM_soil_moisture',
        'LANDSAT_NDWI': 'LANDSAT/LC08/C01/T1_8DAY_NDWI',  # check for cloudcover
        'LANDSAT_NDVI': 'LANDSAT/LC08/C01/T1_8DAY_NDVI',  # check for cloudcover
        'GRIDMET_Precip': 'IDAHO_EPSCOR/GRIDMET',
        'Rainy_days': 'IDAHO_EPSCOR/GRIDMET',
        'MODIS_Day_LST': 'MODIS/006/MOD11A2',  # check for cloudcover
        'MODIS_Terra_NDVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_Terra_EVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_NDWI': 'MODIS/006/MOD09A1',  # cloudcover mask added later
        'MODIS_NDVI': 'MODIS/006/MOD09A1',  # cloudcover mask added later
        'MODIS_LAI': 'MODIS/061/MOD15A2H',
        'MODIS_ET': 'MODIS/006/MOD16A2',  # unit in kg/m2
        'TERRACLIMATE_SR': 'IDAHO_EPSCOR/TERRACLIMATE',
        'GRIDMET_RET': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_max_RH': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_min_RH': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_wind_vel': 'IDAHO_EPSCOR/GRIDMET',  # at 10m
        'GRIDMET_short_rad': 'IDAHO_EPSCOR/GRIDMET',
        'GRIDMET_vap_pres_def': 'IDAHO_EPSCOR/GRIDMET',
        'DAYMET_sun_hr': 'NASA/ORNL/DAYMET_V4',
        'USDA_CDL': 'USDA/NASS/CDL',
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
        'GRIDMET_Precip': 'pr',
        'Rainy_days': 'pr',
        'MODIS_Day_LST': 'LST_Day_1km',
        'MODIS_Terra_NDVI': 'NDVI',
        'MODIS_Terra_EVI': 'EVI',
        'MODIS_NDWI': ['sur_refl_b02', 'sur_refl_b06'],  # bands for NIR and SWIR, respectively
        'MODIS_NDVI': ['sur_refl_b02', 'sur_refl_b01'],  # bands for NIR and SWIR, respectively
        'MODIS_LAI': 'Lai_500m',
        'MODIS_ET': 'ET',
        'TERRACLIMATE_SR': 'ro',  # unit in mm
        'GRIDMET_RET': 'eto',
        'GRIDMET_max_RH': 'rmax',
        'GRIDMET_min_RH': 'rmin',
        'GRIDMET_wind_vel': 'vs',
        'GRIDMET_short_rad': 'srad',
        'GRIDMET_vap_pres_def': 'vpd',
        'DAYMET_sun_hr': 'dayl',
        'USDA_CDL': 'cropland',
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
        'GRIDMET_Precip': 1,
        'Rainy_days': 1,
        'MODIS_Day_LST': 0.02,
        'MODIS_Terra_NDVI': 0.0001,
        'MODIS_Terra_EVI': 0.0001,
        'MODIS_NDWI': 0.0001,
        'MODIS_NDVI': 0.0001,
        'MODIS_LAI': 0.1,
        'MODIS_ET': 0.1,
        'TERRACLIMATE_SR': 1,
        'GRIDMET_RET': 1,
        'GRIDMET_max_RH': 1,
        'GRIDMET_min_RH': 1,
        'GRIDMET_wind_vel': 1,
        'GRIDMET_short_rad': 1,
        'GRIDMET_vap_pres_def': 1,
        'DAYMET_sun_hr': 1,
        'USDA_CDL': 1,
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
        'GRIDMET_Precip': ee.Reducer.sum(),
        'Rainy_days': None,
        'MODIS_Day_LST': ee.Reducer.mean(),
        'MODIS_Terra_NDVI': ee.Reducer.mean(),
        'MODIS_Terra_EVI': ee.Reducer.mean(),
        'MODIS_NDWI': ee.Reducer.mean(),
        'MODIS_NDVI': ee.Reducer.mean(),
        'MODIS_LAI': ee.Reducer.mean(),
        'MODIS_ET': ee.Reducer.sum(),
        'TERRACLIMATE_SR': ee.Reducer.sum(),
        'GRIDMET_RET': ee.Reducer.sum(),
        'GRIDMET_max_RH': ee.Reducer.mean(),
        'GRIDMET_min_RH': ee.Reducer.mean(),
        'GRIDMET_wind_vel': ee.Reducer.mean(),
        'GRIDMET_short_rad': ee.Reducer.mean(),
        'GRIDMET_vap_pres_def': ee.Reducer.mean(),
        'DAYMET_sun_hr': ee.Reducer.mean(),
        'USDA_CDL': ee.Reducer.first(),
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
        'GRIDMET_Precip': datetime(1979, 1, 1),
        'Rainy_days': datetime(1979, 1, 1),
        'MODIS_Day_LST': datetime(2000, 2, 1),
        'MODIS_Terra_NDVI': datetime(2000, 2, 1),
        'MODIS_Terra_EVI': datetime(2000, 2, 1),
        'MODIS_NDWI': datetime(2000, 2, 1),
        'MODIS_NDVI': datetime(2000, 2, 1),
        'MODIS_LAI': datetime(2000, 2, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'TERRACLIMATE_SR': datetime(1958, 1, 1),
        'GRIDMET_RET': datetime(1979, 1, 1),
        'GRIDMET_max_RH': datetime(1979, 1, 1),
        'GRIDMET_min_RH': datetime(1979, 1, 1),
        'GRIDMET_wind_vel': datetime(1979, 1, 1),
        'GRIDMET_short_rad': datetime(1979, 1, 1),
        'GRIDMET_vap_pres_def': datetime(1979, 1, 1),
        'DAYMET_sun_hr': datetime(1980, 1, 1),
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
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
        'GRIDMET_Precip': datetime(2025, 2, 23),
        'Rainy_days': datetime(2025, 2, 23),
        'MODIS_Day_LST': datetime(2023, 8, 29),
        'MODIS_Terra_NDVI': datetime(2023, 8, 13),
        'MODIS_Terra_EVI': datetime(2023, 8, 13),
        'MODIS_NDWI': datetime(2023, 8, 29),
        'MODIS_NDVI': datetime(2023, 8, 29),
        'MODIS_LAI': datetime(2023, 11, 9),
        'MODIS_ET': datetime(2023, 8, 29),
        'TERRACLIMATE_SR': datetime(2022, 12, 1),
        'GRIDMET_RET': datetime(2025, 2, 23),
        'GRIDMET_max_RH': datetime(2025, 2, 23),
        'GRIDMET_min_RH': datetime(2025, 2, 23),
        'GRIDMET_wind_vel': datetime(2025, 2, 23),
        'GRIDMET_short_rad': datetime(2025, 2, 23),
        'GRIDMET_vap_pres_def': datetime(2025, 2, 23),
        'DAYMET_sun_hr': datetime(2023, 12, 31),
        'USDA_CDL': datetime(2023, 1, 1),
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
        'GRIDMET_Precip': datetime(1979, 1, 1),
        'Rainy_days': datetime(1979, 1, 1),
        'MODIS_Day_LST': datetime(2000, 1, 1),
        'MODIS_Terra_NDVI': datetime(2000, 1, 1),
        'MODIS_Terra_EVI': datetime(2000, 1, 1),
        'MODIS_NDWI': datetime(2000, 1, 1),
        'MODIS_NDVI': datetime(2000, 1, 1),
        'MODIS_LAI': datetime(2000, 1, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'TERRACLIMATE_SR': datetime(1958, 1, 1),
        'GRIDMET_RET': datetime(1979, 1, 1),
        'GRIDMET_max_RH': datetime(1979, 1, 1),
        'GRIDMET_min_RH': datetime(1979, 1, 1),
        'GRIDMET_wind_vel': datetime(1979, 1, 1),
        'GRIDMET_short_rad': datetime(1979, 1, 1),
        'GRIDMET_vap_pres_def': datetime(1979, 1, 1),
        'DAYMET_sun_hr': datetime(1980, 1, 1),
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
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
        'GRIDMET_Precip': datetime(2025, 1, 1),
        'Rainy_days': datetime(2025, 1, 1),
        'MODIS_Day_LST': datetime(2024, 1, 1),
        'MODIS_Terra_NDVI': datetime(2024, 1, 1),
        'MODIS_Terra_EVI': datetime(2024, 1, 1),
        'MODIS_NDWI': datetime(2024, 1, 1),
        'MODIS_NDVI': datetime(2024, 1, 1),
        'MODIS_LAI': datetime(2024, 1, 1),
        'MODIS_ET': datetime(2024, 1, 1),
        'TERRACLIMATE_SR': datetime(2023, 1, 1),
        'GRIDMET_RET': datetime(2025, 1, 1),
        'GRIDMET_max_RH': datetime(2025, 1, 1),
        'GRIDMET_min_RH': datetime(2025, 1, 1),
        'GRIDMET_wind_vel': datetime(2025, 1, 1),
        'GRIDMET_short_rad': datetime(2025, 1, 1),
        'GRIDMET_vap_pres_def': datetime(2025, 12, 1),
        'DAYMET_sun_hr': datetime(2025, 1, 1),
        'USDA_CDL': datetime(2024, 1, 1),
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
           year_end_date_dict[data_name]


def cloud_cover_filter(data_name, start_date, end_date, from_bit, to_bit, geometry_bounds):
    """
    Applies cloud cover mask on GEE MODIS data.

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
        ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')
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


def download_soil_datasets(data_name, download_dir, merge_keyword, grid_shape, refraster_westUS=AZ_raster,
                           refraster_gee_merge=GEE_merging_refraster_large_grids,
                           westUS_shape=AZ_shape):
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
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

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
                                       crs='EPSG:26912', ref_raster=refraster_westUS)

        print(f'{data_name} data downloaded and merged')

    else:
        pass


def download_tree_cover_data(data_name, download_dir, merge_keyword, grid_shape, refraster_westUS=AZ_raster,
                             refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=AZ_shape):
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
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

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
                                   crs='EPSG:26912', ref_raster=refraster_westUS)

    print(f'{data_name} data downloaded and merged')


def download_DEM_Slope_data(data_name, download_dir, merge_keyword, grid_shape, refraster_westUS=AZ_raster,
                            refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=AZ_shape,
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
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

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
                                   crs='EPSG:26912', ref_raster=refraster_westUS)

    print(f'{data_name} data downloaded and merged')


# # The download_gee_data_yearly() function isn't fully optimized. Might need to change things at the
# #  get_gee_dict() function. ALso, All it might need modification to download all datasets available in
# the get_gee_dict() function. Not using this function for the current project
def download_gee_data_yearly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
                             refraster_westUS=AZ_raster, refraster_gee_merge=GEE_merging_refraster_large_grids,
                             westUS_shape=AZ_shape):
    """
    Download data (at yearly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GRIDMET_Precip',
        'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_NDVI',
        'MODIS_LAI', 'MODIS_ET', 'TERRACLIMATE_SR',
        'GRIDMET_max_RH', 'GRIDMET_min_RH', 'GRIDMET_wind_vel',
        'GRIDMET_short_rad', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'DAYMET_sun_hr']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 4-12 use (4, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')
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

    for year in year_list:  # first loop for years_list
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

                # This is a check block to see if downloaded datasets are OK
                # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
                # Re-downloading the same data might not have that error
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
                                           crs='EPSG:26912', ref_raster=refraster_westUS)

            print(f'{data_name} yearly data downloaded and merged')

        else:
            pass


def download_gee_data_monthly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
                              use_cpu_while_multidownloading=15, refraster_westUS=AZ_raster,
                              refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=AZ_shape):
    """
    Download data (at monthly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI','GRIDMET_Precip', 'MODIS_Day_LST',
        'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_NDVI', 'MODIS_LAI', 'MODIS_ET',
        'TERRSCLIMATE_SR', 'GRIDMET_max_RH', 'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'Rainy_days',
        'GRIDMET_short_rad', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'DAYMET_sun_hr']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for.
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
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')
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
    data_exclude_list = ['USDA_CDL', 'Field_capacity', 'Bulk_density', 'Organic_carbon_content',
                         'Sand_content', 'Clay_content', 'DEM', 'Effect_precip_DK', 'Tree_cover']

    if data_name not in data_exclude_list:
        # Looping for years_list > months > grids
        for year in year_list:  # first loop for years_list
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

                        elif data_name == 'Rainy_days':
                            precipitation = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                                select(band)

                            def count_rainy_day(img):
                                return img.gt(1)  # boolean image where pixels with precip > 1 mm/day are True (source - WMO)

                            rainy_day = precipitation.map(count_rainy_day)
                            total_rainy_day = rainy_day.reduce(ee.Reducer.sum())
                            download_data = total_rainy_day

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
                                                   crs='EPSG:26912', ref_raster=refraster_westUS)

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
    if '.tif' in file_path:  # only for data downloaded in geotiff format
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
    :param use_cpu: Number of CPU/core (Int) to use for downloading. Default set to 2.

    :return: None.
    """
    # Using ThreadPool() instead of pool() as this is an I/O bound job not CPU bound
    # Using imap() as it completes assigning one task at a time to the ThreadPool()
    # and blocks until each task is complete
    print('######')
    print('Downloading data from GEE..')
    print(f'{cpu_count()} CPUs on this machine. Engaging {use_cpu} CPUs for downloading')
    print('######')

    pool = ThreadPool(use_cpu)
    results = pool.imap(get_data_GEE_saveTopath, download_urls_fp_list)
    pool.close()
    pool.join()


def download_all_gee_data(data_list, download_dir, year_list, month_range,
                          grid_shape_large, use_cpu_while_multidownloading=15,
                          skip_download=False):
    """
    Used to download all gee data together.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['MODIS_Day_LST', 'MODIS_NDWI', 'MODIS_NDVI', 'MODIS_LAI', 'TERRACLIMATE_SR',
        'GRIDMET_Precip', 'GRIDMET_max_RH', 'GRIDMET_min_RH', 'GRIDMET_wind_vel',
        'GRIDMET_short_rad', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'Rainy_days',
        'DAYMET_sun_hr', 'OpenET_ensemble', 'USDA_CDL', 'Field_capacity', 'Bulk_density',
        'Organic_carbon_content', 'Sand_content', 'Clay_content', 'DEM', 'Effect_precip_DK']
        ******************************

    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of years_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_large: File path of larger grids (used in most GEE data download) for which data
                             will be downloaded and mosaiced.
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
                                 'Natural_OpenET_IrrMapper', 'Natural_OpenET_LANID',
                                 'Rainfed_Frac_IrrMapper', 'Rainfed_Frac_LANID', 'USDA_CDL',
                                 'Field_capacity', 'Bulk_density', 'Organic_carbon_content',
                                 'Sand_content', 'Clay_content', 'DEM', 'Effect_precip_DK', 'Tree_cover']:
                # for datasets that needed to be downloaded on monthly scale
                download_gee_data_monthly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                          month_range=month_range, merge_keyword='WestUS_monthly',
                                          refraster_westUS=AZ_raster,
                                          refraster_gee_merge=GEE_merging_refraster_large_grids,
                                          grid_shape=grid_shape_large,
                                          use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'USDA_CDL':
                download_gee_data_yearly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                         month_range=month_range, merge_keyword='WestUS_yearly',
                                         grid_shape=grid_shape_large, refraster_westUS=AZ_raster,
                                         refraster_gee_merge=GEE_merging_refraster_large_grids)

            elif data_name in ['Field_capacity', 'Bulk_density', 'Organic_carbon_content', 'Sand_content',
                               'Clay_content']:
                download_soil_datasets(data_name=data_name, download_dir=download_dir, merge_keyword='WestUS',
                                       grid_shape=grid_shape_large, refraster_westUS=AZ_raster)

            elif data_name == 'DEM':
                download_DEM_Slope_data(data_name=data_name, download_dir=download_dir,
                                        merge_keyword='WestUS', grid_shape=grid_shape_large,
                                        refraster_westUS=AZ_raster,
                                        refraster_gee_merge=GEE_merging_refraster_large_grids,
                                        terrain_slope=False)

            elif data_name == 'Tree_cover':
                download_tree_cover_data(data_name='Tree_cover', download_dir=download_dir,
                                         merge_keyword='WestUS', grid_shape=grid_shape_large,
                                         refraster_westUS=AZ_raster,
                                         refraster_gee_merge=GEE_merging_refraster_large_grids,
                                         westUS_shape=AZ_shape)

    else:
        pass


def download_ssebop_et(years_list, month_range_list, download_dir='../../Data_main/Raster_data/Ssebop_ETa',
                       ssebop_link='https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads/',
                       skip_download=False):
    """
    Download ssebop actual ET data (unit in mm).

    :param years_list: List of years_list for which to download data.
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


def download_all_datasets(year_list, month_range, grid_shape_large,
                          gee_data_list, data_download_dir,
                          skip_download_gee_data=True,
                          use_cpu_while_multidownloading=15):
    """
    Download all GEE datasets.

    :param year_list: List of years_list to download data for. We will use data for [2010, 2015] in the model.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_large: File path of larger grids (used in most GEE data download) for which data
                             will be downloaded and mosaiced.
    :param gee_data_list: List of data to download from GEE. Default set to None, use if skip_download_gee_data=True.
                          Datasets currently downloaded by this code:
                            ['MODIS_Day_LST', 'MODIS_NDWI', 'MODIS_NDVI', 'MODIS_LAI', 'TERRACLIMATE_SR',
                            'GRIDMET_Precip', 'GRIDMET_max_RH', 'GRIDMET_min_RH', 'GRIDMET_wind_vel',
                            'GRIDMET_short_rad', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'Rainy_days',
                            'DAYMET_sun_hr', 'OpenET_ensemble', 'USDA_CDL', 'Field_capacity', 'Bulk_density',
                            'Organic_carbon_content', 'Sand_content', 'Clay_content', 'DEM', 'Effect_precip_DK']
    :param data_download_dir: Directory path to download and save data.
    :param skip_download_gee_data: Set to False if want to download listed data. Default set to True.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    # Data download from GEE
    download_all_gee_data(gee_data_list, download_dir=data_download_dir,
                          year_list=year_list, month_range=month_range,
                          grid_shape_large=grid_shape_large, skip_download=skip_download_gee_data,
                          use_cpu_while_multidownloading=use_cpu_while_multidownloading)
