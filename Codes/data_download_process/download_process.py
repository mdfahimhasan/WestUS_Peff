import os
import ee
import pickle
import zipfile
import requests
import itertools
import numpy as np
import pandas as pd
from glob import glob
from osgeo import gdal
import geopandas as gpd
from Codes.utils.system_ops import makedirs, copy_file
from Codes.utils.vector_ops import add_attr_to_county_fromCSV
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, mosaic_rasters, \
    clip_resample_reproject_raster, mask_raster_by_extent, sum_rasters, mean_rasters, filter_raster_on_threshold

# ee.Authenticate()

# # if ee.Authenticate() doesn't work or gives 'gcloud' error_mae go to the
# 'https://developers.google.com/earth-engine/guides/python_install' link and follow instructions for
# local/remote machine.
# In local machine, install gcloud (if not already installed) and follow the steps from the link to authenticate.
# For remote machine, https://www.youtube.com/watch?v=k-8qFh8EfFA this link was helpful for installing gcloud.
# Couldn't figure out how to authenticate in server pc, will try with new workstation
# If authenticated, no need to run the authentication process again. Just start from ee.initialize()

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, ~2.22 km
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


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
        return download_fname


def extract_data(zip_dir, out_dir, search_by='*.zip', rename_file=True):
    """
    Extract zipped file.

    :param zip_dir: File path of zipped file.
    :param out_dir: File path where data will be extracted.
    :param search_by: Keyword for searching files, default is '*.zip'.
    :param rename_file: True if file rename is required while extracting.

    :return: List of zipped files (can be used for deleting all the files). This function will unzip these files.
    """
    print('Extracting zip files.....')

    makedirs([out_dir])
    all_zipped_files = glob(os.path.join(zip_dir, search_by))

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
        'GPM_PRECIP': 'NASA/GPM_L3/IMERG_MONTHLY_V06',
        'GRIDMET_PRECIP': 'IDAHO_EPSCOR/GRIDMET',
        'PRISM_PRECIP': 'OREGONSTATE/PRISM/AN81m',
        'VIIRS_NDVI': 'NOAA/VIIRS/001/VNP13A1',  # cloudcover mask added later
        'VIIRS_EVI': 'NOAA/VIIRS/001/VNP13A1',  # cloudcover mask added later
        'VIIRS_EVI2': 'NOAA/VIIRS/001/VNP13A1',  # cloudcover mask added later
        'MODIS_Day_LST': 'MODIS/006/MOD11A2',  # check for cloudcover
        'MODIS_Terra_NDVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_Terra_EVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_NDWI': 'MODIS/006/MOD09A1',  # cloudcover mask added later
        'MODIS_LAI': 'MODIS/006/MCD15A3H',
        'MODIS_ET': 'MODIS/006/MOD16A2',  # unit in kg/m2
        'FAO_ACTUAL_ET': 'FAO/WAPOR/2/L1_AETI_D',
        'FAO_EVAPORATION': 'FAO/WAPOR/2/L1_E_D',
        'FLDAS_BF_GW_RO': 'NASA/FLDAS/NOAH01/C/GL/M/V001',
        'FLDAS_SM': 'NASA/FLDAS/NOAH01/C/GL/M/V001',
        'USDA_CDL': 'USDA/NASS/CDL',  # Recommending not to use GEE. Download data from direct USDA link
        'GPW_Pop': 'CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density'
    }
    gee_band_dict = {
        'SMAP_SM': 'ssm',
        'LANDSAT_NDWI': 'NDWI', 'NDWI'
                                'LANDSAT_NDVI': 'NDVI',
        'GPM_PRECIP': 'precipitation',
        'GRIDMET_PRECIP': 'pr',
        'PRISM_PRECIP': 'ppt',
        'VIIRS_NDVI': 'NDVI',
        'VIIRS_EVI': 'EVI',
        'VIIRS_EVI2': 'EVI2',
        'MODIS_Day_LST': 'LST_Day_1km',
        'MODIS_Terra_NDVI': 'NDVI',
        'MODIS_Terra_EVI': 'EVI',
        'MODIS_NDWI': ['sur_refl_b02', 'sur_refl_b06'],  # bands for NIR and SWIR, respectively.
        'MODIS_LAI': 'Lai',
        'MODIS_ET': 'ET',
        'FAO_ACTUAL_ET': 'L1_AETI_D',
        'FAO_EVAPORATION': 'L1_E_D',
        'FLDAS_BF_GW_RO': 'Qsb_tavg',
        'FLDAS_SM': 'SoilMoi00_10cm_tavg',
        'USDA_CDL': 'cropland',
        'GPW_Pop': 'unwpp-adjusted_population_density'
    }
    gee_scale_dict = {
        'SMAP_SM': 1,
        'LANDSAT_NDWI': 1,
        'LANDSAT_NDVI': 1,
        'GPM_PRECIP': 1,
        'GRIDMET_PRECIP': 1,
        'PRISM_PRECIP': 1,
        'VIIRS_NDVI': 0.0001,
        'VIIRS_EVI': 0.0001,
        'VIIRS_EVI2': 0.0001,
        'MODIS_Day_LST': 0.02,
        'MODIS_Terra_NDVI': 0.0001,
        'MODIS_Terra_EVI': 0.0001,
        'MODIS_NDWI': 0.0001,
        'MODIS_LAI': 0.1,
        'MODIS_ET': 0.1,
        'FAO_ACTUAL_ET': 0.1,
        'FAO_EVAPORATION': 0.1,
        'FLDAS_BF_GW_RO': 1,
        'FLDAS_SM': 1,
        'USDA_CDL': 1,
        'GPW_Pop': 1
    }

    aggregation_dict = {
        'SMAP_SM': ee.Reducer.sum(),
        'LANDSAT_NDWI': ee.Reducer.mean(),
        'LANDSAT_NDVI': ee.Reducer.mean(),
        'GPM_PRECIP': ee.Reducer.sum(),
        'GRIDMET_PRECIP': ee.Reducer.sum(),
        'PRISM_PRECIP': ee.Reducer.sum(),
        'VIIRS_NDVI': ee.Reducer.mean(),
        'VIIRS_EVI': ee.Reducer.mean(),
        'VIIRS_EVI2': ee.Reducer.mean(),
        'MODIS_Day_LST': ee.Reducer.mean(),
        'MODIS_Terra_NDVI': ee.Reducer.mean(),
        'MODIS_Terra_EVI': ee.Reducer.mean(),
        'MODIS_NDWI': ee.Reducer.mean(),
        'MODIS_LAI': ee.Reducer.mean(),
        'MODIS_ET': ee.Reducer.sum(),
        'FAO_ACTUAL_ET': ee.Reducer.sum(),
        'FAO_EVAPORATION': ee.Reducer.sum(),
        'FLDAS_BF_GW_RO': ee.Reducer.sum(),
        'FLDAS_SM': ee.Reducer.sum(),
        'USDA_CDL': ee.Reducer.first(),
        'GPW_Pop': ee.Reducer.first()
    }

    return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name]


def cloud_cover_filter(data_name, start_date, end_date, from_bit, to_bit, geometry_bounds):
    """
    Applies cloud cover mask on GEE data.

    :param data_name: Data Name.
           Valid dataset include- ['MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'VIIRS_NDVI',
           'VIIRS_EVI', 'VIIRS_EVI2'].
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
        :param Image.
        :return Cloud-masked image.
        """
        ee.Initialize()
        global qc_img
        if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
            qc_img = img.select('DetailedQA')
        elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
            qc_img = img.select('VI_Quality')
        elif data_name == 'MODIS_NDWI':
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

    elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
        image = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1').filterDate(start_date, end_date).\
            filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked

    elif data_name == 'MODIS_NDWI':
        image = ee.ImageCollection('MODIS/061/MOD09A1').filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked


def download_gee_data_for_grow_season\
                (data_name, download_dir, year_list, merge_keyword, refraster=WestUS_raster,
                 grid_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'):
    """
    process and download data from GEE based on varying growing season. This code will mosaic data for different growing
    season together. So there will be visible difference in data. Consider smoothening the grids used for downloading.

    **Saving as a reference code. Will not use in this project. Rather data will be downloaded for each month
    (for selected years) and then merged together based on growing season to make the data smoother.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for, e.g. [2000, 2005, 2010, 2015].
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WesternUS'/'Conus'.
    :param refraster: Reference raster to use for merging downloaded datasets.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.

    :return: File path of downloaded raster data.
    """
    ee.Initialize()
    makedirs([download_dir])

    downloaded_raster_dict = {}

    grow_season_dict = {'Apr01-Oct31': [4, 10], 'Jan01-Dec31': [1, 12], 'Jul01-Aug31': [7, 8], 'Mar01-Nov30': [3, 11],
                        'May01-Sep30': [5, 9]}  # A dictionary for assigning growing season months

    for year in year_list:
        grids = gpd.read_file(grid_shape)
        grids = grids.sort_values(by='grid_no', ascending=True)
        growing_season = grids['GrowSeason']
        grid_geometry = grids['geometry']
        grid_no = grids['grid_no']

        start_year, end_year, start_month, end_month = None, None, None, None

        for grid, grow_period, serial in zip(grid_geometry, growing_season, grid_no):
            roi = grid.bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            # Selecting month ranges based on grid files' growing season periods
            if grow_period == 'Apr01-Oct31':
                start_year = year
                end_year = year
                start_month, end_month = grow_season_dict['Apr01-Oct31'][0], grow_season_dict['Apr01-Oct31'][1]

            elif grow_period == 'Jan01-Dec31':
                start_year = year
                end_year = year + 1
                start_month, end_month = grow_season_dict['Jan01-Dec31'][0], 1  # up to January 1st of next year

            elif grow_period == 'Jul01-Aug31':
                start_year = year
                end_year = year
                start_month, end_month = grow_season_dict['Jul01-Aug31'][0], grow_season_dict['Jul01-Aug31'][1]

            elif grow_period == 'Mar01-Nov30':
                start_year = year
                end_year = year
                start_month, end_month = grow_season_dict['Mar01-Nov30'][0], grow_season_dict['Mar01-Nov30'][1]

            elif grow_period == 'May01-Sep30':
                start_year = year
                end_year = year
                start_month, end_month = grow_season_dict['May01-Sep30'][0], grow_season_dict['May01-Sep30'][1]

            start_date = ee.Date.fromYMD(start_year, start_month, 1)
            end_date = ee.Date.fromYMD(end_year, end_month + 1, 1)
            data, band, multiply_scale, reducer = get_gee_dict(data_name)

            if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band).\
                    reduce(reducer).multiply(multiply_scale).toFloat()

            elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
                download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band).\
                    reduce(reducer).multiply(multiply_scale).toFloat()

            elif data_name == 'MODIS_NDWI':
                nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0])\
                    .reduce(reducer).multiply(multiply_scale).toFloat()
                swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1])\
                    .reduce(reducer).multiply(multiply_scale).toFloat()
                download_data = nir.subtract(swir).divide(nir.add(swir))
            elif data_name == 'GPW_Pop':
                start_date = ee.Date.fromYMD(year, 1, 1)  # GPW population dataset's data starts at
                end_date = ee.Date.fromYMD(year, 12, 31)
                download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date).\
                    filterBounds(gee_extent).reduce(reducer).toFloat()
            else:
                download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date).\
                    filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

            data_url = download_data.getDownloadURL({'name': data_name,
                                                     'crs': 'EPSG:4269',  # NAD83
                                                     'scale': 2000,  # in meter. equal to ~0.02 deg
                                                     'region': gee_extent,
                                                     'format': 'GEO_TIFF'})
            key_word = data_name
            local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(serial)}.tif')
            print('Downloading', local_file_name, '.....')
            r = requests.get(data_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)

        mosaic_name = f'{data_name}_{year}.tif'
        mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
        makedirs([mosaic_dir])
        downloaded_arr, downloaded_raster = mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster,
                                                           search_by=f'*{year}*.tif', nodata=no_data_value)
        print('Downloaded Data Merged')
        downloaded_raster_dict[mosaic_name] = downloaded_raster

    return downloaded_raster_dict


def download_gee_data_monthly(data_name, download_dir, year_list, month_range, merge_keyword, refraster=WestUS_raster,
                              grid_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'):
    """
    Download data (at monthly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WesternUS'/'Conus'.
    :param refraster: Reference raster to use for merging downloaded datasets.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.

    :return: File path of downloaded raster data.
    """
    ee.Initialize()
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
    for year in year_list:  # first loop for years
        for month in month_list:  # second loop for months
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = ee.Date.fromYMD(year, month + 1, 1)

            for grid_sr, geometry in zip(grid_no, grid_geometry):  # third loop for grids
                roi = geometry.bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Filtering/processing datasets with data ranges, cloudcover, geometry, band, reducer, scale
                if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                    download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band).\
                        reduce(reducer). \
                        multiply(multiply_scale).toFloat()

                elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
                    download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band).\
                        reduce(reducer).multiply(multiply_scale).toFloat()

                elif data_name == 'MODIS_NDWI':
                    nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]).\
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]).\
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    download_data = nir.subtract(swir).divide(nir.add(swir))
                elif data_name == 'GPW_Pop':
                    start_date = ee.Date.fromYMD(year, 1, 1)  # GPW population dataset's data starts at
                    end_date = ee.Date.fromYMD(year, 12, 31)
                    # filterBounds are not necessary, added it to reduce processing extent
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date).\
                        filterBounds(gee_extent).reduce(reducer).toFloat()
                else:
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date).\
                        filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                data_url = download_data.getDownloadURL({'name': data_name,
                                                         'crs': 'EPSG:4269',  # NAD83
                                                         'scale': 2000,  # in meter. equal to ~0.02 deg
                                                         'region': gee_extent,
                                                         'format': 'GEO_TIFF'})
                key_word = data_name
                local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')
                print('Downloading', local_file_name, '.....')
                r = requests.get(data_url, allow_redirects=True)
                open(local_file_name, 'wb').write(r.content)

            mosaic_name = f'{data_name}_{year}_{month}.tif'
            mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
            makedirs([mosaic_dir])
            search_by = f'*{year}_{month}*.tif'
            mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster, search_by=search_by,
                           nodata=no_data_value)
            print(f'{data_name} monthly data downloaded and merged')


def download_gee_data_yearly(data_name, download_dir, year_list, month_range, merge_keyword, refraster=WestUS_raster,
                              grid_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'):
    """
    Download data (at yearly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 4-12 use (4, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WesternUS'/'Conus'.
    :param refraster: Reference raster to use for merging downloaded datasets.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.

    :return: File path of downloaded raster data.
    """
    ee.Initialize()
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years
        start_date = ee.Date.fromYMD(year, month_range[0], 1)
        end_date = ee.Date.fromYMD(year, month_range[1] + 1, 1)

        for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
            roi = geometry.bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            # Filtering/processing datasets with data ranges, cloudcover, geometry, band, reducer, scale
            if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band).\
                    reduce(reducer). \
                    multiply(multiply_scale).toFloat()

            elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
                download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band).\
                    reduce(reducer).multiply(multiply_scale).toFloat()

            elif data_name == 'MODIS_NDWI':
                nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]).\
                    reduce(reducer).multiply(multiply_scale).toFloat()
                swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]).\
                    reduce(reducer).multiply(multiply_scale).toFloat()
                download_data = nir.subtract(swir).divide(nir.add(swir))
            elif data_name == 'GPW_Pop':
                start_date = ee.Date.fromYMD(year, 1, 1)  # GPW population dataset's data starts at
                end_date = ee.Date.fromYMD(year, 12, 31)
                # filterBounds are not necessary, added it to reduce processing extent
                download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date).\
                    filterBounds(gee_extent).reduce(reducer).toFloat()
            else:
                download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date).\
                    filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

            data_url = download_data.getDownloadURL({'name': data_name,
                                                     'crs': 'EPSG:4269',  # NAD83
                                                     'scale': 2000,  # in meter. equal to ~0.02 deg
                                                     'region': gee_extent,
                                                     'format': 'GEO_TIFF'})
            key_word = data_name
            local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')
            print('Downloading', local_file_name, '.....')
            r = requests.get(data_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)

        mosaic_name = f'{data_name}_{year}.tif'
        mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
        makedirs([mosaic_dir])
        mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster, search_by=f'*{year}*.tif',
                       nodata=no_data_value)
        print(f'{data_name} yearly data downloaded and merged')


def download_all_gee_data(data_list, download_dir, year_list, month_range, refraster=WestUS_raster,
                          skip_download=False,
                          grid_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'):
    """
    Used to download all gee data together.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM']
    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param refraster: Reference raster to use for merging downloaded datasets.
    :param skip_download: Set to True to skip download.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.

    :return: None
    """
    if not skip_download:
        for data_name in data_list:
            data_download_dir = os.path.join(download_dir, data_name)
            makedirs([data_download_dir])

            # for datasets that needed to be downloaded on monthly scale
            if data_name in ['SMAP_SM', 'GPM_PRECIP', 'MODIS_NDWI', 'MODIS_ET']:
                download_gee_data_monthly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                          month_range=month_range, merge_keyword='WestUS_monthly',
                                          refraster=refraster, grid_shape=grid_shape)
            # for datasets that needed to be downloaded on yearly scale
            elif data_name in ['GPW_Pop']:
                download_gee_data_yearly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                          month_range=month_range, merge_keyword='WestUS',
                                          refraster=refraster, grid_shape=grid_shape)
    else:
        pass


def download_ssebop_et(years_list, month_range_list, download_dir='../../Data_main/Raster_data/Ssebop_monthly_ETa',
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


def download_all_datasets(year_list, month_range, gee_data_list=None, skip_download_gee_data=True,
                          skip_download_ssebop_data=True):
    """
    Download all GEE datasets and ssebop data.

    :param year_list: List of years to download data for. We will use data for [2010, 2015] in the model.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param gee_data_list: List of data to download from GEE. Default set to None, use if skip_download_gee_data=True.
                          Datasets currently used in the model:
                          ['MODIS_NDWI', 'GPM_PRECIP', 'MODIS_Day_LST', 'PRISM_PRECIP']
    :param skip_download_gee_data: Set to False if want to download listed data. Default set to True.
    :param skip_download_ssebop_data: Set to False if want to download ssebop data. Default set to True.

    :return: None.
    """
    download_all_gee_data(gee_data_list, download_dir='../../Data_main/Raster_data',
                          year_list=year_list, month_range=month_range, refraster=WestUS_raster,
                          skip_download=skip_download_gee_data,
                          grid_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp')

    download_ssebop_et(year_list, month_range_list=[1, 12],
                       download_dir='../../Data_main/Raster_data/Ssebop_monthly_ETa',
                       skip_download=skip_download_ssebop_data)


def mask_cdl_to_WestUS(cdl_dir='../../Data_main/Raster_data/USDA_CDL/CONUS_original_30m',
                       westus_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_30m',
                       westUS_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS.shp',
                       processing_shp_dir='../../Data_main/Raster_data/USDA_CDL/processing_shapes',
                       nodata=no_data_value, try_maximum_occurrence_approach=False):
    """
    Crop and resample cdl raster fro Western US and create 2000 & 2005 cdl data.

    CDL data download link : https://www.nass.usda.gov/Research_and_Science/Cropland/Release/

    :param cdl_dir: Directory path of original (whole US) CDL datasets.
    :param processed_dir: File path of directory to save processed data.
    :param westUS_shape: Western US shapefile.
    :param try_maximum_occurrence_approach: Set to True if want to create 2000 and 2005 cdl data based on maximum
                                            occurrence approach. Maximum occurrence approach assigns the most frequent
                                            cdl data value from 2008-2011 to 2000 and 2005 cds raster.
                                            try_maximum_occurrence_approach=False uses 2008 cdl data as 200 and 2005
                                            cdl rasters.

    :return: Cropped and resampled cdl rasters of 2000, 2005, 2010, 2015.
    """
    # cdl_US = glob(os.path.join(cdl_dir, '*.tif'))
    # processing_bounds = glob(os.path.join(processing_shp_dir, '*.shp'))

    # New directory inside westus_dir ro save processed cropland data
    cropland_dir = os.path.join(westus_dir, 'cropland')
    makedirs([cropland_dir])

    cdl_2008_arr, cdl_2009_arr, cdl_2010_arr, cdl_2011_arr, cdl_file, shape = None, None, None, None, None, None
    # Loop for masking cdl raster to Western US extent. Using 4 bounding boxes to minimize memory load.
    # for cdl_ras in cdl_US:
    #     for shape in processing_bounds:
    #         raster_name = os.path.splitext(os.path.basename(cdl_ras))[0]
    #         shape_name = os.path.splitext(os.path.basename(shape))[0]
    #         name = raster_name + '_' + shape_name + '.tif'
    #         mask_raster_by_extent(input_raster=cdl_ras, ref_file=shape, output_dir=westus_dir,
    #                               raster_name=name, invert=False, crop=True, nodata=0)


    # converting to crop vs non-crop and developed vs non-developed data
    noncrop_classes = [0, 58, 59, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 121, 122, 123, 124, 131, 141, 142,
                       143, 152, 176, 190, 195]  # 0 is no data value

    cdl_westus = glob(os.path.join(westus_dir, '*.tif'))
    for cdl_ras in cdl_westus:
            # crop vs non-crop raster  # if array value in noncrop list assigns 0, otherwise assigns 1
            # cdl data has no_data value set to 0
            # data_arr = np.where(ref_arr == 0, 1, ref_arr)
            cdl_arr, cdl_file = read_raster_arr_object(cdl_ras, change_dtype=False)  # the data type will be converted to float
            crop_arr = np.where(~np.isin(cdl_arr, noncrop_classes), 1, 0)

            crop_raster_name = os.path.splitext(os.path.basename(cdl_ras))[0] + '_crop.tif'
            crop_raster_path = os.path.join(cropland_dir, crop_raster_name)
            # saving cropland rasters
            write_array_to_raster(raster_arr=crop_arr, raster_file=cdl_file, transform=cdl_file.transform,
                                  output_path=crop_raster_path, nodata=0)

            # developed vs non-developed  # if array value is not within 121, 122, 123, and 124 assigns 0
            ##########
            dev_arr = np.where(cdl_arr == 121, 1, cdl_arr)  # 1: Developed/Open Space
            dev_arr = np.where(cdl_arr == 122, 2, dev_arr)  # 2: Developed/Low Intensity
            dev_arr = np.where(cdl_arr == 123, 3, dev_arr)  # 3: Developed/Medium Intensity
            dev_arr = np.where(cdl_arr == 124, 4, dev_arr)  # 4: Developed/High Intensity

        # westus_cdl = clip_resample_reproject_raster(cdl_ras, westUS_shape, keyword='WestUS',
        #                                             output_raster_dir=processed_dir, clip_and_resample=True,
        #                                             targetaligned=False, resample_algorithm='near',
        #                                             resolution=model_res, crs='EPSG:4269',
        #                                             output_datatype=gdal.GDT_Byte)
    #     if '2008' in westus_cdl:
    #         cdl_2008_arr, cdl_file = read_raster_arr_object(westus_cdl)
    #         shape = cdl_2008_arr.shape
    #         cdl_2008_arr = cdl_2008_arr.flatten()
    #     elif '2009' in westus_cdl:
    #         cdl_2009_arr = read_raster_arr_object(westus_cdl, get_file=False).flatten()
    #     elif '2010' in westus_cdl:
    #         cdl_2010_arr = read_raster_arr_object(westus_cdl, get_file=False).flatten()
    #     elif '2011' in westus_cdl:
    #         cdl_2011_arr = read_raster_arr_object(westus_cdl, get_file=False).flatten()
    #
    # if try_maximum_occurrence_approach:
    #     # setting maximum occurring crop value of 2008-2011 cdl array as values of 2000 and 2005 cdl array
    #     max_arr = np.stack([cdl_2008_arr, cdl_2009_arr, cdl_2010_arr, cdl_2011_arr], axis=0)
    #     new_arr = np.zeros(cdl_2008_arr.shape)
    #
    #     # this loop takes each column of stacked array. np.bincount() counts the number of occurance of each value in
    #     # the selected column. Then, np.argmax() selects the value which occurs most frequently. Then, we paste the
    #     # selected value to a new array
    #     for i in range(max_arr.shape[1]):
    #         selected_arr = max_arr[:, i]
    #         max_occurred_value = np.argmax(np.bincount(list(selected_arr)))  # # # improve this. try to avoid loop
    #         new_arr[i] = max_occurred_value
    #
    #     max_arr = new_arr.reshape(shape)
    #
    # else:
    #     max_arr = cdl_2008_arr.reshape(shape)
    #
    # output_2000_cdl_data = os.path.join(processed_dir, 'WestUS_2000_30m_cdls.tif')
    # output_2005_cdl_data = os.path.join(processed_dir, 'WestUS_2005_30m_cdls.tif')
    #
    # write_array_to_raster(max_arr, raster_file=cdl_file, transform=cdl_file.transform,
    #                       output_path=output_2000_cdl_data)
    # write_array_to_raster(max_arr, raster_file=cdl_file, transform=cdl_file.transform,
    #                       output_path=output_2005_cdl_data)


mask_cdl_to_WestUS(cdl_dir='../../Data_main/Raster_data/USDA_CDL/CONUS_original_30m',
                   westus_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_30m',
                   westUS_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS.shp',
                   processing_shp_dir='../../Data_main/Raster_data/USDA_CDL/processing_shapes',
                   try_maximum_occurrence_approach=False)

def preprocess_cdl_data(cdl_dir='../../Data_main/Raster_data/USDA_CDL/WestUS',
                        processed_dir='../../Data_main/Raster_data/USDA_CDL/WestUS',
                        ref_raster=WestUS_raster, skip_processing=False):
    """
    Preprocess (reclassify) cdl data for Western US. Reclassifies cdl to crop vs non-crop and developed vs
    non-developed datasets.

    Developed classes: 121, 122, 123, 124
    non-crop classes : 58, 59, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 121, 122, 123, 124, 131, 141, 142, 143,
                       152, 176, 190, 195
    details in: https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL

    :param cdl_dir: Directory path of original (whole US) CDL datasets.
    :param processed_dir: File path of directory to save processed data.
    :param ref_raster: File path of reference raster. Default set to WesternUS 2km reference raster.
    :param skip_processing: Set to True to skip processing.

    :return: Dictionaries of file paths of processed crop vs non-crop and developed vs non-developed datasets.
    """
    crop_data_dict, developed_data_dict = None, None

    if not skip_processing:
        makedirs([processed_dir])
        crop_cdl_and_create_2000_2005_cdl_raster()  # cropping cdl rasters and creating 2000 and 2005 cdl rasters

        crop_data_dict = {}
        developed_data_dict = {}

        cdl_US = glob(os.path.join(cdl_dir, '*WestUS*.tif'))
        year_list = ['2000', '2005', '2010', '2015']
        filtered_cdl_list = []

        for cdl in cdl_US:
            year = os.path.basename(cdl).split('_')[1]
            if year in year_list:
                filtered_cdl_list.append(cdl)

        for cdl_ras in filtered_cdl_list:
            # converting to crop vs non-crop and developed vs non-developed data
            noncrop_classes = [58, 59, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 121, 122, 123, 124, 131, 141, 142,
                               143, 152, 176, 190, 195]

            year = os.path.splitext(os.path.basename(cdl_ras))[0].split('_')[1]
            cdl_arr = read_raster_arr_object(cdl_ras, get_file=False)
            ref_arr, ref_file = read_raster_arr_object(ref_raster)

            # crop vs non-crop raster  # if array value in noncrop list assigns 0, otherwise assigns 1
            # cdl data has no_data value set to 0
            data_arr = np.where(ref_arr == 0, 1, ref_arr)
            crop_arr = np.where(~np.isin(cdl_arr, noncrop_classes) & (data_arr == 1) & (cdl_arr > 0), 1, ref_arr)

            crop_raster_path = os.path.join(processed_dir, f'USDA_cropland_{year}.tif')
            crop_output_file = write_array_to_raster(crop_arr, ref_file, ref_file.transform, crop_raster_path)

            # developed vs non-developed  # if array value is not within 121, 122, 123, and 124 assigns 0
            dev_arr = np.where(cdl_arr == 121, 1, ref_arr)  # 1: Developed/Open Space
            dev_arr = np.where(cdl_arr == 122, 2, dev_arr)  # 2: Developed/Low Intensity
            dev_arr = np.where(cdl_arr == 123, 3, dev_arr)  # 3: Developed/Medium Intensity
            dev_arr = np.where(cdl_arr == 124, 4, dev_arr)  # 4: Developed/High Intensity

            dev_raster_path = os.path.join(processed_dir, f'USDA_developed_{year}.tif')
            dev_output_file = write_array_to_raster(dev_arr, ref_file, ref_file.transform, dev_raster_path)

            crop_output_file = copy_file(crop_output_file, copy_dir='../../Data_main/Compiled_data')
            data_name = crop_output_file[crop_output_file.rfind(os.sep) + 1:crop_output_file.rfind('.')]
            crop_data_dict[data_name] = crop_output_file

            dev_output_file = copy_file(dev_output_file, copy_dir='../../Data_main/Compiled_data')
            data_name = dev_output_file[dev_output_file.rfind(os.sep) + 1:dev_output_file.rfind('.')]
            developed_data_dict[data_name] = dev_output_file

        pickle.dump(crop_data_dict, open('../../Data_main/Compiled_data/crop_data_dict.pkl', mode='wb+'))
        pickle.dump(developed_data_dict, open('../../Data_main/Compiled_data/developed_land_data_dict.pkl', mode='wb+'))

    else:
        crop_data_dict = pickle.load(open('../../Data_main/Compiled_data/crop_data_dict.pkl', mode='rb'))
        developed_data_dict = pickle.load(open('../../Data_main/Compiled_data/developed_land_data_dict.pkl', mode='rb'))

    return crop_data_dict, developed_data_dict


def process_irrigated_landuse_data(input_dir='../../Data_main/Raster_data/Irrigated_agriculture',
                                   input_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                                   output_dir='../../Data_main/Compiled_data', skip_processing=False):
    """
    Process irrigated cropland data from Xie et al. 2021. Two sets of rasters are generated. First: Irrigated area
    raster with values 0 and 1 (1 menas irrigated); second: Irrigated fraction data (values in fraction from 0 or 1,
    higher fraction means higher irrigated lands in that 2-km pixel).

    :param input_dir: Filepath of input data directory.
    :param input_shape: Filepath of ROI shapefile.
    :param output_dir: Filepath of output data directory.
    :param skip_processing: Bool. Set to True is want to skip processing.

    :return: A dictionary of processed dataset filepaths. The dictionary has years (e.g., 2015) as keys.
    """
    irrigated_area_dict = None
    irrigated_frac_dict = None

    if not skip_processing:
        print('Processing Irrigated Agriculture dataset....')
        input_raster = glob(os.path.join(input_dir, '*.tif'))

        irrigation_area_dict = {}  # a dictionary to store irrigated area data (values in 0 or 1, 1 means irrigated)
        irrigation_frac_dict = {}  # a dictionary to store irrigated fraction data (values in fraction from 0 or 1)

        for raster in input_raster:
            if any([i in raster for i in ['clipped', 'resample', 'irrigated']]):
                os.remove(raster)  # to remove raster previously processed
            else:
                # The dataset is 30m resolution CONUS scale and huge to clip and resample at the same time
                # Resampling to model resolution first and then clipping. "average" resampling calculates the
                # fraction of irrigated area (30m pixel) in each 2-km pixel.

                # # Processing irrigated area data to fraction of irrigated area data
                resampled_raster = clip_resample_reproject_raster(
                    input_raster=raster, input_shape=input_shape, keyword='resampled', resolution=model_res,
                    output_raster_dir=input_dir, clip_and_resample=False, clip=False, resample=True,
                    resample_algorithm='average')

                clipped_raster = clip_resample_reproject_raster(
                    input_raster=resampled_raster, input_shape=input_shape, keyword='clipped', resolution=model_res,
                    output_raster_dir=input_dir, clip_and_resample=True, clip=False, resample=False,
                    targetaligned=False)  # targetAlignedPixels=False to ensure equal row-columns like other dataset
                # copying to compiled directory
                year = os.path.splitext(os.path.basename(clipped_raster))[0][-4:]
                output_name = f'irrigated_agri_frac_{year}'
                copied_file = copy_file(input_dir_file=clipped_raster, copy_dir=output_dir, rename=output_name)
                irrigation_frac_dict[year] = copied_file

                # # Processing irrigated area data to presence of irrigated area data
                # Using threshold to choose irrigated area
                output_name = f'irrigated_agri_{year}'
                output_fp = os.path.join(input_dir, f'{output_name}.tif')
                filter_raster_on_threshold(input_raster=clipped_raster, output_raster=output_fp, threshold_value1=0.3,
                                           assign_value=1)

                # copying to compiled directory
                copied_file = copy_file(input_dir_file=output_fp, copy_dir=output_dir, rename=None)
                irrigation_area_dict[year] = copied_file

        print('Processed Irrigated Agriculture dataset')
        pickle.dump(irrigation_frac_dict, open('../../Data_main/Compiled_data/irrigated_area_frac_dict.pkl', mode='wb+'))
        pickle.dump(irrigation_area_dict, open('../../Data_main/Compiled_data/irrigated_area_dict.pkl', mode='wb+'))

    else:
        irrigated_frac_dict = pickle.load(open('../../Data_main/Compiled_data/irrigated_area_frac_dict.pkl', mode='rb'))
        irrigated_area_dict = pickle.load(open('../../Data_main/Compiled_data/irrigated_area_dict.pkl', mode='rb'))

    return irrigated_frac_dict, irrigated_area_dict


def process_ssebop_data(years=(2005, 2010, 2015), ssebop_dir='../../Data_main/Raster_data/Ssebop_monthly_ETa',
                        output_dir_ssebop='../../Data_main/Raster_data/Ssebop_monthly_ETa/WesternUS',
                        ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) Ssebop ET.

    :param years: Tuple/list of years for which ssebop data was downloaded.
    :param ssebop_dir: Directory file path of downloaded ssebop datasets.
    :param output_dir_ssebop: File path of directory to save summed ET for each year at Western US extent.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000389.
    :param skip_processing: Set to True if want to skip ssebop processing.

    :return: A dictionary with ssebop data file paths.
    """
    if not skip_processing:
        interim_dir_for_clip = os.path.join(ssebop_dir, 'interim_dir_clip')
        interim_dir_for_sum = os.path.join(ssebop_dir, 'interim_dir_sum')
        makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_ssebop])

        ssebop_processing_dict = {}

        # Growing season shapefile
        grow_season_Apr_Oct_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Apr_Oct.shp'
        grow_season_Jan_Dec_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jan_Dec.shp'
        grow_season_Jul_Aug_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jul_Aug.shp'
        grow_season_Mar_Nov_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Mar_Nov.shp'
        grow_season_May_Sep_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_May_Sep.shp'

        # Assigning growing season shapefiles in a dictionary
        grow_season_shape_dict = {'Apr_Oct': grow_season_Apr_Oct_shp, 'Jan_Dec': grow_season_Jan_Dec_shp,
                                  'Jul_Aug': grow_season_Jul_Aug_shp, 'Mar_Nov': grow_season_Mar_Nov_shp,
                                  'May_sep': grow_season_May_Sep_shp}

        # Creating a dictionary of growing seasons' month span
        month_dict = {'Apr_Oct': range(4, 11), 'Jan_Dec': range(1, 13), 'Jul_Aug': range(7, 9),
                      'Mar_Nov': range(3, 12), 'May_sep': range(5, 10)}

        for year in years:  # first loop for years
            print(f'Processing Ssebop data for {year}...')
            ssebop_datasets = glob(os.path.join(ssebop_dir, f'*{year}*.tif'))  # monthly ssebop datasets for each year

            for season in grow_season_shape_dict.keys():  # second loop for growing season zones
                growing_season_shape = grow_season_shape_dict[season]
                months = list(month_dict[season])

                ssebop_clipped_list = []
                for monthly_ssebop in ssebop_datasets:  # third loop for selected monthly ssebop data for each year
                    ssebop_month = os.path.basename(monthly_ssebop).split('.')[0][-2:]  # month extracted as string
                    ssebop_month = int(ssebop_month)  # month converted to integer

                    if ssebop_month in months:
                        clipped_ssebop = \
                            clip_resample_reproject_raster(input_raster=monthly_ssebop,
                                                           input_shape=growing_season_shape,
                                                           keyword=' ', output_raster_dir=interim_dir_for_clip,
                                                           clip=False, resample=False, clip_and_resample=True,
                                                           targetaligned=True, resample_algorithm='near',
                                                           resolution=resolution)

                        ssebop_clipped_list.append(clipped_ssebop)

                # Summing raster for each growing season
                summed_output_for_season = os.path.join(interim_dir_for_sum, f'ssebop_{year}_' + season + '.tif')
                sum_rasters(raster_list=ssebop_clipped_list, raster_dir=None, output_raster=summed_output_for_season,
                            ref_raster=ssebop_clipped_list[0])

            # mosaicing ssebop ET for all growing seasons in a year
            yearly_summed_ssebop = f'SSEBOP_ET_{year}.tif'
            search_by = f'*{year}*.tif'
            _, mosaiced_ssebop = mosaic_rasters(input_dir=interim_dir_for_sum, output_dir=output_dir_ssebop,
                                                raster_name=yearly_summed_ssebop, ref_raster=ref_raster,
                                                search_by=search_by, resolution=resolution)

            # Copying mosaiced ssebop ET to compiled data folder and saving in a dictionary to load later
            copy_file(mosaiced_ssebop, copy_dir='../../Data_main/Compiled_data')
            ssebop_processing_dict[yearly_summed_ssebop[:yearly_summed_ssebop.rfind('.')]] = mosaiced_ssebop
            pickle.dump(ssebop_processing_dict, open('../../Data_main/Compiled_data/ssebop_dict.pkl', mode='wb+'))
    else:
        ssebop_processing_dict = pickle.load(open('../../Data_main/Compiled_data/ssebop_dict.pkl', mode='rb'))

        return ssebop_processing_dict


def convert_prism_data_to_tif(input_dir, output_dir, keyword='prism_precip'):
    """
    Convert prism rainfall/temperature datasets from .bil format to GeoTiff format.

    Download PRISM datasets directly from  'https://prism.oregonstate.edu/recent/'

    :param input_dir: Directory path of prism data in .bil format.
    :param output_dir: Directory path of converted (.tif) prism data.
    :param keyword: keyword to add before processed datasets.

    :return: None.
    """
    makedirs([output_dir])
    prism_datasets = glob(os.path.join(input_dir, '*.bil'))

    for data in prism_datasets:
        year_month = os.path.basename(data).split('_')[-2]
        output_name = keyword + '_' + year_month + '.tif'
        output_file = os.path.join(output_dir, output_name)
        gdal.Translate(destName=output_file, srcDS=data, format='GTiff', outputType=gdal.GDT_Float32,
                       outputSRS='EPSG:4269')


def process_prism_data(prism_bil_dir, prism_dir, output_dir_prism, years=(2005, 2010, 2015), keyword='prism_precip',
                       ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) Prism Precipitation.

    :param prism_dir: Directory file path of downloaded prism datasets in .bil format.
    :param prism_dir: Directory file path of prism datasets converted to tif format.
    :param output_dir_prism: File path of directory to save summed/mean prism precipitation/temperature data for each
                             year at Western US extent.
    :param years: Tuple/list of years for which prism data was downloaded.
    :param keyword: keyword to add before processed datasets.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000389.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: A dictionary with prism precip data file paths.
    """
    if not skip_processing:
        convert_prism_data_to_tif(input_dir=prism_bil_dir, output_dir=prism_dir, keyword=keyword)

        interim_dir_for_clip = os.path.join(prism_dir, 'interim_dir_clip')

        # Creating additional folders/directories needed for processing (clip/sum/mean)
        if 'precip' in keyword:
            interim_dir_for_sum = os.path.join(prism_dir, 'interim_dir_sum')
            makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_prism])
        elif 'temp' in keyword:
            interim_dir_for_mean = os.path.join(prism_dir, 'interim_dir_mean')
            makedirs([interim_dir_for_clip, interim_dir_for_mean, output_dir_prism])


        prism_processing_dict = {}

        # Growing season shapefile
        grow_season_Apr_Oct_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Apr_Oct.shp'
        grow_season_Jan_Dec_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jan_Dec.shp'
        grow_season_Jul_Aug_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jul_Aug.shp'
        grow_season_Mar_Nov_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Mar_Nov.shp'
        grow_season_May_Sep_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_May_Sep.shp'

        # Assigning growing season shapefiles in a dictionary
        grow_season_shape_dict = {'Apr_Oct': grow_season_Apr_Oct_shp, 'Jan_Dec': grow_season_Jan_Dec_shp,
                                  'Jul_Aug': grow_season_Jul_Aug_shp, 'Mar_Nov': grow_season_Mar_Nov_shp,
                                  'May_sep': grow_season_May_Sep_shp}

        # Creating a dictionary of growing seasons' month span
        month_dict = {'Apr_Oct': range(4, 11), 'Jan_Dec': range(1, 13), 'Jul_Aug': range(7, 9),
                      'Mar_Nov': range(3, 12), 'May_sep': range(5, 10)}

        for year in years:  # first loop for years
            print(f'Processing {keyword} data for {year}...')
            prism_datasets = glob(os.path.join(prism_dir, f'*{year}*.tif'))  # monthly prism datasets for each year

            for season in grow_season_shape_dict.keys():  # second loop for growing season zones
                growing_season_shape = grow_season_shape_dict[season]
                months = list(month_dict[season])

                prism_clipped_list = []
                for monthly_prism in prism_datasets:  # third loop for selected monthly prism data for each year
                    prism_month = os.path.basename(monthly_prism).split('.')[0][-2:]  # month extracted as string
                    prism_month = int(prism_month)  # month converted to integer

                    if prism_month in months:
                        clipped_prism = \
                            clip_resample_reproject_raster(input_raster=monthly_prism,
                                                           input_shape=growing_season_shape,
                                                           keyword=' ', output_raster_dir=interim_dir_for_clip,
                                                           clip=False, resample=False, clip_and_resample=True,
                                                           targetaligned=True, resample_algorithm='near',
                                                           resolution=resolution)

                        prism_clipped_list.append(clipped_prism)

                if 'precip' in keyword:
                    # Summing raster for each growing season
                    summed_output_for_season = os.path.join(interim_dir_for_sum, f'prism_{year}_' + season + '.tif')
                    sum_rasters(raster_list=prism_clipped_list, raster_dir=None, output_raster=summed_output_for_season,
                                ref_raster=prism_clipped_list[0])

                    # mosaicing prism precip for all growing seasons in a year
                    yearly_prism = f'{keyword}_{year}.tif'
                    search_by = f'*{year}*.tif'
                    _, mosaiced_prism = mosaic_rasters(input_dir=interim_dir_for_sum, output_dir=output_dir_prism,
                                                       raster_name=yearly_prism, ref_raster=ref_raster,
                                                       search_by=search_by, resolution=resolution)

                elif 'temp' in keyword:
                    # Calculating mean of raster for each growing season
                    mean_output_for_season = os.path.join(interim_dir_for_mean, f'prism_{year}_' + season + '.tif')
                    mean_rasters(raster_list=prism_clipped_list, raster_dir=None, output_raster=mean_output_for_season,
                                 ref_raster=prism_clipped_list[0])

                    # mosaicing prism temperature for all growing seasons in a year
                    yearly_prism = f'{keyword}_{year}.tif'
                    search_by = f'*{year}*.tif'
                    _, mosaiced_prism = mosaic_rasters(input_dir=interim_dir_for_mean, output_dir=output_dir_prism,
                                                       raster_name=yearly_prism, ref_raster=ref_raster,
                                                       search_by=search_by, resolution=resolution)

            # Copying mosaiced prism precip to compiled data folder and saving in a dictionary to load later
            copy_file(mosaiced_prism, copy_dir='../../Data_main/Compiled_data')
            prism_processing_dict[yearly_prism[:yearly_prism.rfind('.')]] = mosaiced_prism
            pickle.dump(prism_processing_dict, open('../../Data_main/Compiled_data/prism_dict.pkl', mode='wb+'))
    else:
        prism_processing_dict = pickle.load(open('../../Data_main/Compiled_data/prism_dict.pkl', mode='rb'))

        return prism_processing_dict


def run_all_preprocessing(skip_cdl_processing=False, skip_ssebop_processing=False, skip_irrigatedCrop_processing=False,
                          skip_prism_processing=False):
    """
    Run all preprocessing steps.

    :param skip_cdl_processing: Set True if want to skip cdl data preprocessing.
    :param skip_ssebop_processing: Set True if want to skip ssebop data preprocessing.
    :param skip_irrigatedCrop_processing: Set True if want to skip Irrigated cropland data preprocessing.
    :param skip_prism_processing: Set True if want to skip prism (precipitation and temperature) data preprocessing.

    :return: Dictionaries of crop and developed land use data file paths
    """
    crop_data_dict, developed_data_dict = preprocess_cdl_data(skip_processing=skip_cdl_processing)
    ssebop_dict = process_ssebop_data(skip_processing=skip_ssebop_processing)
    irrigated_crop_dict = process_irrigated_landuse_data(skip_processing=skip_irrigatedCrop_processing)
    prism_precip_dict = process_prism_data(prism_bil_dir='../../Data_main/Raster_data/PRISM_PRECIP/bil_format',
                                           prism_dir='../../Data_main/Raster_data/PRISM_PRECIP/tif_format',
                                           output_dir_prism='../../Data_main/Raster_data/PRISM_PRECIP/WestUS',
                                           keyword='prism_precip', skip_processing=skip_prism_processing)

    prism_temp_dict = process_prism_data(prism_bil_dir='../../Data_main/Raster_data/PRISM_PRECIP/bil_format',
                                         prism_dir='../../Data_main/Raster_data/PRISM_PRECIP/tif_format',
                                         output_dir_prism='../../Data_main/Raster_data/PRISM_PRECIP/WestUS',
                                         keyword='prism_temp', skip_processing=skip_prism_processing)

    return crop_data_dict, developed_data_dict, ssebop_dict, irrigated_crop_dict, prism_precip_dict, prism_temp_dict


def compile_USGS_WaterUse_data(data_dir='../../Data_main/USGS_water_use_data', search_by='*201[0-5]*.xlsx',
                               county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county_projected.shp',
                               output_csv='../../Data_main/USGS_water_use_data/WestUS_county_WaterUse.csv',
                               output_shape='../../Data_main/Compiled_data/County_total_WaterUse.shp',
                               skip_compiling=True, ref_raster=WestUS_raster):
    """
    Compile county-wide water use data for Western US from USGS dataset.

    :param data_dir: Directory path of yearly groundwater data excels.
    :param search_by: Search pattern for yearly data excels. Default set to '*201[0-5]*.xlsx' for selecting 2010 and
                      2015 data only.
    :param county_shape: File path of Western US county shapefile location.
    :param output_csv: Filepath to save output csv.
    :param output_shape: Filepath of output county shapefile added with water use data.
    :param skip_compiling: Set to False to compile pumping data. Default set to True (data already compiled).
    :param ref_raster: Model reference raster filepath.

    :return: Compiled USGS pumping data csv.
    """
    if not skip_compiling:
        import warnings
        warnings.simplefilter('ignore')  # adding to remove warning from excel opening

        print('Compiling USGS Water Use Data...')

        wateruse_data = glob(os.path.join(data_dir, search_by))
        county_df = gpd.read_file(county_shape)

        # Counting Irrigated crop and developed area in each county
        countyID_arr, county_file = read_raster_arr_object('../../Data_main/Compiled_data/Western_US_countyID.tif')
        irrig_arr, _ = read_raster_arr_object('../../Data_main/Compiled_data/irrigated_agri_2015.tif')
        developed_arr_2015, _ = read_raster_arr_object('../../Data_main/Compiled_data/USDA_developed_2015.tif')
        #### modify for irrigation and developed of 2010 area data when adding 2010
        ref_arr, _ = read_raster_arr_object(ref_raster)

        # Array of pixels which are irrigated or developed
        landUse_arr_2015 = np.where((irrig_arr == 1) | (developed_arr_2015 > 0), 1, ref_arr)
        # Counting how many pixels in each county has irrigated and developed pixels
        unique, count = np.unique(countyID_arr[(landUse_arr_2015 == 1) & (~np.isnan(countyID_arr))], return_counts=True)
        count_dict = dict(zip(unique, count))

        county_wateruse = pd.DataFrame()
        for data in wateruse_data:
            wateruse_df = pd.read_excel(data, sheet_name='CountyData', engine='openpyxl')
            wateruse_df = wateruse_df[['COUNTY', 'STATE', 'COUNTYFIPS', 'FIPS', 'YEAR', 'IR-WSWFr', 'PT-WSWFr',
                                       'TO-WGWFr', 'TO-WSWFr', 'TO-WFrTo']]

            joined_df = county_df.merge(wateruse_df, left_on='fips', right_on='FIPS', how='inner')
            # adding irrigated and developed pixels count in the dataframe
            joined_df['irrig_dev_pixels'] = None
            joined_df['irrig_dev_pixels'] = joined_df['fips'].map(count_dict)
            county_wateruse = pd.concat([county_wateruse, joined_df])

        # converting groundwater withdrawal from Mgal/day to mm/year.
        area_single_pixel = (2.22 * 2.22) * (1000 * 1000)  # area of a pixel in m2
        county_wateruse['gw_withdrawal'] = 1000 * (1e6 * county_wateruse['TO-WGWFr'] * 0.00378541 * 365 /
                                                   (county_wateruse['irrig_dev_pixels'] * area_single_pixel))
        county_wateruse['sw_withdrawal'] = 1000 * (1e6 * (county_wateruse['TO-WSWFr'] - county_wateruse['PT-WSWFr'])
                                                   * 0.00378541 * 365 /
                                                   (county_wateruse['irrig_dev_pixels'] * area_single_pixel))
        county_wateruse['total_withdrawal'] = 1000 * (1e6 * (county_wateruse['TO-WSWFr'] - county_wateruse['PT-WSWFr'])
                                                      * 0.00378541 * 365 /
                                                      (county_wateruse['irrig_dev_pixels'] * area_single_pixel))

        county_wateruse = county_wateruse[['NAME', 'fips', 'YEAR', 'gw_withdrawal',
                                           'sw_withdrawal', 'total_withdrawal']]
        county_wateruse_df = county_wateruse.rename(columns={'NAME': 'Name', 'fips': 'fips', 'YEAR': 'Year'})
        # Saving as csv
        GW_use_usgs_csv = output_csv
        county_wateruse_df.to_csv(GW_use_usgs_csv, index=False)

        # Converting to shapefile
        add_attr_to_county_fromCSV(input_shapefile='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county.shp',
                                   attr_csv_data=GW_use_usgs_csv,
                                   output_shapefile=output_shape, year_filter=2015,  ## modify it later
                                   columns_to_keep=('Name', 'fips', 'Year', 'gw_withdrawal', 'sw_withdrawal',
                                                    'total_withdrawal'))
        print('USGS Water Use Data Compiled\n')

    else:
        print('Loading USGS Water Use Data...\n')
        GW_use_usgs_csv = output_csv

    return GW_use_usgs_csv



