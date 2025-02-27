import os
import ee
import sys
import time
import requests
import geopandas as gpd
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, clip_resample_reproject_raster, mosaic_rasters_from_directory

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


def get_openet_gee_dict(data_name):
    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    gee_data_dict = {
        'OpenET_ensemble': ['OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
                            'projects/openet/assets/ensemble/conus/gridmet/monthly/provisional'],
        'Irrig_crop_OpenET_IrrMapper': ['OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
                                        'projects/openet/assets/ensemble/conus/gridmet/monthly/provisional'],
        'IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
        'Irrigation_Frac_IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
    }

    gee_band_dict = {
        'OpenET_ensemble': 'et_ensemble_mad',
        'Irrig_crop_OpenET_IrrMapper': 'et_ensemble_mad',  # unit in mm, monthly total
        'IrrMapper': 'classification',
        'Irrigation_Frac_IrrMapper': 'classification',
    }

    gee_scale_dict = {
        'OpenET_ensemble': 1,
        'Irrig_crop_OpenET_IrrMapper': 1,
        'IrrMapper': 1,
        'Irrigation_Frac_IrrMapper': 1
    }

    aggregation_dict = {
        'OpenET_ensemble': ee.Reducer.mean(),           # monthly data; doesn't matter whether use mean() or sum() as reducer. Change for yearly data download if needed.
        'Irrig_crop_OpenET_IrrMapper': ee.Reducer.sum(),
        'IrrMapper': ee.Reducer.max(),
        'Irrigation_Frac_IrrMapper': ee.Reducer.max()
    }

    # # Note on start date and end date dictionaries
    # The start and end dates have been set based on what duration of data can be downloaded.
    # They may not exactly match with the data availability in GEE
    # In most cases the end date is shifted a month later to cover the end month's data

    month_start_date_dict = {
        'OpenET_ensemble': datetime(1985, 1, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(1985, 1, 1),
        'IrrMapper': datetime(1985, 1, 1),
        'Irrigation_Frac_IrrMapper': datetime(1986, 1, 1)
    }

    month_end_date_dict = {
        'OpenET_ensemble': datetime(2022, 12, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(2023, 12, 1),
        'IrrMapper': datetime(2024, 1, 1),
        'Irrigation_Frac_IrrMapper': datetime(2024, 1, 1)
    }

    year_start_date_dict = {
        'OpenET_ensemble': datetime(1985, 1, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(1985, 1, 1),
        'IrrMapper': datetime(1985, 1, 1),
        'Irrigation_Frac_IrrMapper': datetime(1985, 1, 1)
    }

    year_end_date_dict = {
        'OpenET_ensemble': datetime(2023, 1, 1),
        'Irrig_crop_OpenET_IrrMapper': datetime(2023, 12, 1),
        'IrrMapper': datetime(2024, 1, 1),
        'Irrigation_Frac_IrrMapper': datetime(2024, 1, 1)
    }

    return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name], \
           month_start_date_dict[data_name], month_end_date_dict[data_name], year_start_date_dict[data_name], \
           year_end_date_dict[data_name]


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


def download_openet_ensemble(download_dir, year_list, month_range, merge_keyword, grid_shape,
                             use_cpu_while_multidownloading=15, refraster_westUS=AZ_raster,
                             refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=AZ_shape):
    """
    Download openET ensemble data (at monthly scale) from GEE.

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
    global openet_asset, data_url

    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, 'OpenET_ensemble')
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_openet_gee_dict('OpenET_ensemble')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

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

            # selecting open vs provisional data asset in GEE
            # openET 1985-1999 data is provisional and 2000 to upfront data in open in GEE
            # selecting appropriate OpenET GEE asset based on year
            if year >= 2000:
                openet_asset = data[0]
            elif year == 1999 and month in [10, 11, 12]:
                openet_asset = data[0]
            else:
                openet_asset = data[1]

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= month_start_range) and (end_date_dt <= month_end_range):
                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    download_data = ee.ImageCollection(openet_asset).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to
                    # retry failed connections
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data_url = download_data.getDownloadURL({'name': 'OpenET_ensemble',
                                                                          'crs': 'EPSG:4269',  # NAD83
                                                                          'scale': 2200,  # in meter. equal to ~0.02 deg
                                                                          'region': gee_extent,
                                                                          'format': 'GEO_TIFF'})
                            break  # if successful, exit the loop
                        except ee.EEException as e:
                            if attempt < max_retries - 1:
                                time.sleep(5)  # wait for 5 seconds before retrying
                                continue
                            else:
                                print(f"Failed to get data_url for year={year}, month={month}, grid={grid_sr}: {e}")
                                data_url = None

                    key_word = 'OpenET_ensemble'
                    local_file_path = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')

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

                mosaic_name = f'OpenET_ensemble_{year}_{month}.tif'
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

                print('OpenET_ensemble monthly data downloaded and merged')

            else:
                print(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_Irr_frac_from_IrrMapper_yearly(data_name, download_dir, year_list, grid_shape,
                                            use_cpu_while_multidownloading=15):
    """
    Download IrrMapper Irrigated fraction data (at 2km scale) at yearly scale from GEE for 11 states in the Western US
    WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM for 2000-2020.
    *** For downloading irrigated fraction data for ND, SD, OK, KS, NE, and TX use download_Irr_frac_from_LANID_yearly()
    function.

    ########################
    # READ ME (for Irrigation Data)
    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 1986-2023,
    whereas LANID data consists datasets these 11 states and of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2020.
    Our target year is 1999-2020. We are using IrrMapper data as irrigated cropland data for 1999-2020 for the available
    11 states' data (also it's available in GEE already). For the other 06 states, we clipped LANID data and compiled
    them as a GEE asset for 1999-2020. In summary, for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states)
    the irrigated field data comes from IrrMapper and for ND, SD, OK, KS, NE, and TX the irrigated field data comes from
    LANID. Also, adding AIM-HPA data with LANID to pick up irrigated fields that are unseen/undetected in LANID data.
    ########################
.
    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrigation_Frac_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    global data_url

    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, _, _, year_start_range, year_end_range = get_openet_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years_list
        print('********')
        print(f'Getting data urls for year={year}.....')
        start_date_dt = datetime(year, 1, 1)
        end_date_dt = datetime(year, 12, 31)

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) and (end_date_dt <= year_end_range):
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

                # Getting Data URl for each grid from GEE
                # The GEE connection gets disconnected sometimes, therefore, we adding the try-except block to
                # retry failed connections
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                              'crs': 'EPSG:4269',  # NAD83
                                                              'scale': 2200,  # in meter. equal to ~0.02 deg
                                                              'region': gee_extent,
                                                              'format': 'GEO_TIFF'})
                        break  # if successful, exit the loop
                    except ee.EEException as e:
                        if attempt < max_retries - 1:
                            time.sleep(5)  # wait for 5 seconds before retrying
                            continue
                        else:
                            print(f"Failed to get data_url for year={year}, grid={grid_sr}: {e}")
                            data_url = None

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


def download_Irr_CropET_from_OpenET_IrrMapper_monthly(data_name, download_dir, year_list, month_range, grid_shape,
                                                      scale=2200, use_cpu_while_multidownloading=15):
    """
    Download irrigated cropET data (at monthly scale) from OpenET GEE by filtering ET data with irrigated field data
    from IrrMapper,available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 1985-2023


    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrig_crop_OpenET_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    global openet_asset, data_url

    ee.Initialize(project='ee-fahim', opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting IrrMapper and OpenET dataset information required for downloading from GEE
    et_data, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
    _, _ = get_openet_gee_dict(data_name)

    irr_data, irr_band, irr_multiply_scale, irr_reducer, _, _, _, _ = get_openet_gee_dict('IrrMapper')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    for year in year_list:  # first loop for years_list
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

            # selecting open vs provisional data asset in GEE
            # openET 1985-1999 data is provisional and 2000 to upfront data in open in GEE
            # selecting appropriate OpenET GEE asset based on year
            if year >= 2000:
                openet_asset = et_data[0]
            elif year == 1999 and month in [10, 11, 12]:
                openet_asset = et_data[0]
            else:
                openet_asset = et_data[1]

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= et_month_start_range) and (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(openet_asset)
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
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                          'crs': 'EPSG:4269',  # NAD83
                                                                          'scale': scale,
                                                                          # in meter. equal to ~0.02 deg
                                                                          'region': gee_extent,
                                                                          'format': 'GEO_TIFF'})
                            break  # if successful, exit the loop
                        except ee.EEException as e:
                            if attempt < max_retries - 1:
                                time.sleep(5)  # wait for 5 seconds before retrying
                                continue
                            else:
                                print(f"Failed to get data_url for year={year}, month={month}, grid={grid_sr}: {e}")
                                data_url = None

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


def download_openET_data(data_list, download_dir, year_list, month_range,
                         grid_shape_for_2km_ensemble, grid_shape_for30m_irrmapper,
                         GEE_merging_refraster=GEE_merging_refraster_large_grids,
                         westUS_refraster=AZ_raster, westUS_shape=AZ_shape,
                         use_cpu_while_multidownloading=15, skip_download=False):
    """
    Used to download openET datasets from GEE.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['Irrig_crop_OpenET_IrrMapper', 'Irrigation_Frac_IrrMapper', 'OpenET_ensemble']
        ******************************

    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of years_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_for_2km_ensemble: File path of larger grids to download data for Western US.
    :param grid_shape_for30m_irrmapper: File path of smaller grids to download data for IrrMapper extent and cropET from
                                        openET (this datasets are processed at 30m res in GEE, so smaller grids are
                                        required).
    :param GEE_merging_refraster: Reference raster to mosaic openET ensemble 2km dataset.
    :param westUS_refraster: Western US reference raster.
    :param westUS_shape: Western US shapefile.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param skip_download: Set to True to skip download.

    :return: None
    """
    if not skip_download:

        for data_name in data_list:

            if data_name == 'OpenET_ensemble':
                download_openet_ensemble(download_dir=download_dir, year_list=year_list,
                                         month_range=month_range, merge_keyword='WestUS_monthly',
                                         grid_shape=grid_shape_for_2km_ensemble,
                                         use_cpu_while_multidownloading=15, refraster_westUS=westUS_refraster,
                                         refraster_gee_merge=GEE_merging_refraster,
                                         westUS_shape=westUS_shape)


            elif data_name == 'Irrig_crop_OpenET_IrrMapper':
                download_Irr_CropET_from_OpenET_IrrMapper_monthly(data_name=data_name, download_dir=download_dir,
                                                                  year_list=year_list, month_range=month_range,
                                                                  grid_shape=grid_shape_for30m_irrmapper, scale=2200,
                                                                  use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrigation_Frac_IrrMapper':
                download_Irr_frac_from_IrrMapper_yearly(data_name=data_name, download_dir=download_dir,
                                                        year_list=year_list, grid_shape=grid_shape_for30m_irrmapper,
                                                        use_cpu_while_multidownloading=use_cpu_while_multidownloading)
    else:
        pass


def download_all_openET_datasets(year_list, month_range,
                                 grid_shape_for_2km_ensemble,
                                 grid_shape_for30m_irrmapper,
                                 openET_data_list, data_download_dir,
                                 GEE_merging_refraster=GEE_merging_refraster_large_grids,
                                 westUS_refraster=AZ_raster, westUS_shape=AZ_shape,
                                 skip_download_OpenET_data=True,
                                 use_cpu_while_multidownloading=15):
    """
    Download all GEE datasets.

    :param year_list: List of years_list to download data for. We will use data for [2010, 2015] in the model.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_for_2km_ensemble: File path of larger grids to download data for Western US.
    :param grid_shape_for30m_irrmapper: File path of smaller grids to download data for IrrMapper extent and cropET from
                                        openET (this datasets are processed at 30m res in GEE, so smaller grids are
                                        required).
    :param openET_data_list: List of data to download by processing openET dataset.
                             Datasets currently downloaded by this code:
                                ['Irrig_crop_OpenET_IrrMapper', 'Irrigation_Frac_IrrMapper', 'OpenET_ensemble']
    :param data_download_dir: Directory path to download and save data.
    :param GEE_merging_refraster: Reference raster to mosaic openET ensemble 2km dataset.
    :param westUS_refraster: Western US reference raster.
    :param westUS_shape: Western US shapefile.
    :param skip_download_OpenET_data: Set to False if want to download listed data. Default set to True.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    # OpenET data download from GEE
    download_openET_data(openET_data_list, download_dir=data_download_dir,
                         year_list=year_list, month_range=month_range,
                         grid_shape_for_2km_ensemble=grid_shape_for_2km_ensemble,
                         grid_shape_for30m_irrmapper=grid_shape_for30m_irrmapper,
                         GEE_merging_refraster=GEE_merging_refraster,
                         westUS_refraster=westUS_refraster, westUS_shape=westUS_shape,
                         skip_download=skip_download_OpenET_data,
                         use_cpu_while_multidownloading=use_cpu_while_multidownloading)
