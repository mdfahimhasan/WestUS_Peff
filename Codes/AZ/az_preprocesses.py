import os
import re
import sys
import datetime
import numpy as np
from glob import glob
from osgeo import gdal
import rasterio as rio
from rasterio.merge import merge
from rasterio.enums import Resampling

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, \
    clip_resample_reproject_raster, sum_rasters, mean_rasters


no_data_value = -9999
model_res = 2000  # in m
AZ_shape = '../../Data_main/AZ_files/ref_files/AZ.shp'
AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'
GEE_merging_refraster_large_grids = '../../Data_main/AZ_files/ref_files/AZ_gee_merge_ref_raster.tif'
GEE_merge_ref_raster_30m =  '../../Data_main/AZ_files/ref_files/AZ_gee_merge_ref_raster_30m.tif'



def mosaic_rasters_list(input_raster_list, output_dir, raster_name, ref_raster, dtype=None,
                        resampling_method='nearest', mosaicing_method='first', resolution=None,
                        nodata=no_data_value):
    """
    Mosaics a list of input rasters.


    :param input_raster_list: A list of input rasters to merge/mosaic.
    :param output_dir: Output raster directory.
    :param raster_name: Output raster name.
    :param ref_raster: Reference raster filepath.
    :param dtype: Output raster data type. Default set to None.
    :param resampling_method: Resampling method. Default set to 'nearest'. Can also take 'bilinear'. Currently can use
                              only these two resampling methods.
    :param mosaicing_method: Mosaicing method. Can be 'first' or 'max' or 'min'. Default set to 'first'.
    :param resolution: Resolution of the output raster. Default set to None to use the first input raster's resolution.
    :param nodata: no_data_value set as -9999.

    :return: Mosaiced raster array and filepath of mosaiced raster.
    """
    raster_file_list = []  # a list to store raster file information
    for raster in input_raster_list:
        arr, file = read_raster_arr_object(raster)
        raster_file_list.append(file)

    # setting resampling method
    if resampling_method == 'nearest':
        resampling_method = Resampling.nearest
    elif resampling_method == 'bilinear':
        resampling_method = Resampling.bilinear

    # reference raster
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    # merging
    if resolution is None:  # will use first input raster's resolution
        merged_arr, out_transform = merge(raster_file_list, bounds=ref_file.bounds,
                                          resampling=resampling_method, method=mosaicing_method,
                                          nodata=nodata)

    else:  # will use input resolution
        merged_arr, out_transform = merge(raster_file_list, bounds=ref_file.bounds, res=(resolution, resolution),
                                          resampling=resampling_method, method=mosaicing_method, nodata=nodata)
    # nodata operation
    merged_arr = np.where(ref_arr == 0, merged_arr, ref_arr)
    merged_arr = merged_arr.squeeze()

    # saving output
    makedirs([output_dir])
    out_raster = os.path.join(output_dir, raster_name)
    write_array_to_raster(raster_arr=merged_arr, raster_file=ref_file, transform=out_transform,
                          output_path=out_raster, nodata=nodata, ref_file=ref_raster, dtype=dtype)

    return merged_arr, out_raster


def merge_GEE_data_patches_IrrMapper_extents(year_list, input_dir_irrmapper, merged_output_dir,
                                             merge_keyword, monthly_data=True, ref_raster=GEE_merge_ref_raster_30m,
                                             az_shape=AZ_shape, skip_processing=False):
    """
    Merge/mosaic downloaded GEE data for project extent.

    :param year_list: Tuple/list of years_list for which data will be processed.
    :param input_dir_irrmapper: Input directory filepath of datasets at IrrMapper extent.
    :param merged_output_dir: Output directory filepath to save merged data.
    :param merge_keyword: Keyword to use while merging. Foe example: 'Rainfed_Frac', 'Irrigated_crop_OpenET', etc.
    :param monthly_data: Boolean. If False will look/search for yearly data patches. Default set to True to look for
                         monthly datasets.
    :param ref_raster: Reference raster to use in merging. Default set to Western US reference raster.
    :param az_shape: Arizona state shapefile.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    :return: None.
    """
    if not skip_processing:
        interim_output_dir = os.path.join(merged_output_dir, 'interim')
        makedirs([interim_output_dir, merged_output_dir])

        if monthly_data:  # for datasets that are monthly
            month_list = list(range(1, 13))

            for year in year_list:
                for month in month_list:
                    search_by = f'*{year}_{month}_*.tif'

                    # collecting all raster chunks
                    total_raster_list = glob(os.path.join(input_dir_irrmapper, search_by))

                    if len(total_raster_list) > 0:  # to only merge for years_list and months when data is available
                        merged_raster_name = f'{merge_keyword}_{year}_{month}.tif'
                        _, merged_raster = \
                            mosaic_rasters_list(input_raster_list=total_raster_list, output_dir=interim_output_dir,
                                                raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                                resampling_method='nearest', mosaicing_method='first',
                                                resolution=None, nodata=no_data_value)

                        clip_resample_reproject_raster(input_raster=merged_raster, input_shape=az_shape,
                                                       output_raster_dir=merged_output_dir,
                                                       raster_name=os.path.basename(merged_raster),
                                                       clip_and_resample=True,
                                                       resolution=model_res,
                                                       crs='EPSG:26912',
                                                       ref_raster=ref_raster)

                        print(f'{merge_keyword} data merged for year {year}, month {month}')

        else:  # for datasets that are yearly
            for year in year_list:
                search_by = f'*{year}_*.tif'

                # collecting all raster chunks
                total_raster_list = glob(os.path.join(input_dir_irrmapper, search_by))

                if len(total_raster_list) > 0:  # to only merge for years_list and months when data is available
                    merged_raster_name = f'{merge_keyword}_{year}.tif'
                    _, merged_raster = \
                        mosaic_rasters_list(input_raster_list=total_raster_list, output_dir=interim_output_dir,
                                            raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                            resampling_method='nearest', mosaicing_method='first',
                                            resolution=None, nodata=no_data_value)

                    clip_resample_reproject_raster(input_raster=merged_raster, input_shape=az_shape,
                                                   output_raster_dir=merged_output_dir,
                                                   raster_name=os.path.basename(merged_raster),
                                                   clip_and_resample=True,
                                                   resolution=model_res,
                                                   crs='EPSG:26912',
                                                   ref_raster=ref_raster)

                    print(f'{merge_keyword} data merged for year {year}')
    else:
        pass


def classify_irrigated_cropland(irrigated_fraction_dir,
                                irrigated_cropland_output_dir,
                                skip_processing=False):
    """
    Classifies irrigated cropland using irrigated fraction data.

    :param irrigated_fraction_dir: Input directory path for irrigated fraction data.
    :param irrigated_cropland_output_dir: Output directory path for classified irrigated cropland data.
    :param skip_processing: Set to True if want to skip classifying irrigated and rainfed cropland data.

    :return: None
    """
    if not skip_processing:
        makedirs([irrigated_cropland_output_dir])

        ############################

        # # Irrigated
        # A 2km pixel with >2% irr fraction was used to classify as irrigated
        irrigated_frac_threshold_for_irrigated_class = 0.02

        # Classifying those data with defined threshold
        years = list(range(1985, 2025))         # 1985-2024

        for year in years:
            print(f'Classifying irrigated cropland data for year {year}')

            irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')
            irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

            # classification using defined irrigated fraction
            irrigated_cropland = np.where((irrig_arr > irrigated_frac_threshold_for_irrigated_class), 1, -9999)

            # saving classified data
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir,
                                                            f'Irrigated_cropland_{year}.tif')
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster,
                                  dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
    else:
        pass


def filter_irrigated_cropET_with_irrigated_cropland(irrigated_cropland_dir,
                                                    irrigated_cropET_input_dir,
                                                    irrigated_cropET_output_dir,
                                                    skip_processing=False):
    """
    Filter Irrigated and Rainfed cropET data by rainfed and irrigated cropland, respectively.

    ** The downloaded Irrigated and Rainfed cropET data from GEE is not fully filtered for rainfed and irrigated
    cropland because in some pixels there are some rainfed and some irrigated fields. So, we first classify rainfed and
    irrigated cropland by rainfed and irrigation fraction threshold (detail in classify_irrigated_rainfed_cropland()
    func), then apply the cropland filters to extract cropET on the purely rainfed and irrigated cropland
    pixels, respectively.

    :param irrigated_cropland_dir: Input directory filepath of irrigated cropland data.
    :param irrigated_cropET_input_dir: Input directory filepath of raw irrigated cropET data.
    :param irrigated_cropET_output_dir: Output directory filepath of filtered irrigated cropET data.
    :param skip_processing: Set to True if want to skip filtering irrigated and rainfed cropET data.

    :return: None.
    """
    if not skip_processing:
        makedirs([irrigated_cropET_output_dir])

        # cropET datasets have been extracted from openET for the following years_list and months only
        years_to_filter_irrig_cropET = list(range(1985, 2025))          # 1985-2024

        months_to_filter_cropET = list(range(1, 13))

        for year in years_to_filter_irrig_cropET:

            print(f'Filtering irrigated cropET data for year {year}...')

            # pure irrigated cropland filtered by using irrigated fraction threshold (irrig frac > 0.02)
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)

            for month in months_to_filter_cropET:

                # # applying irrigated cropland filter to get cropET at purely irrigated pixels
                irrigated_cropET_data = glob(os.path.join(irrigated_cropET_input_dir, f'*{year}_{month}*.tif'))[0]
                irrigated_cropET_arr, irrigated_cropET_file = read_raster_arr_object(irrigated_cropET_data)

                # applying the filter
                irrigated_cropET_arr[np.isnan(irrigated_cropland_arr)] = -9999

                filtered_output_raster = os.path.join(irrigated_cropET_output_dir,
                                                      f'Irrigated_cropET_{year}_{month}.tif')
                write_array_to_raster(raster_arr=irrigated_cropET_arr, raster_file=irrigated_cropET_file,
                                      transform=irrigated_cropET_file.transform, output_path=filtered_output_raster)
    else:
        pass


def sum_GridMET_precip_yearly_data(year_list, input_gridmet_monthly_dir, output_dir_yearly,
                                   skip_processing=False):
    """
    Process (sum for Western US extent) GRIDMET Precipitation for a year and water year.

    :param year_list: Tuple/list of years_list for which data will be processed.
    :param input_gridmet_monthly_dir: Directory file path of downloaded gridmet precip monthly datasets.
    :param output_dir_yearly: File path of directory to save summed precip for each year at Western US extent.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_yearly])

        for yr in year_list:  # first loop for years_list
            print(f'summing GRIDMET precip data for year {yr}...')
            gridmet_datasets = glob(os.path.join(input_gridmet_monthly_dir, f'*{yr}*.tif'))

            # Summing raster for each year
            summed_output_for_year = os.path.join(output_dir_yearly, f'GRIDMET_Precip_{yr}.tif')
            sum_rasters(raster_list=gridmet_datasets, raster_dir=None, output_raster=summed_output_for_year,
                        ref_raster=gridmet_datasets[0])

    else:
        pass


def sum_GridMET_RET_yearly_data(input_RET_monthly_dir, output_dir_RET_yearly, output_dir_RET_growing_season,
                                year_list, skip_processing=False):
    """
    Process GridMET RET datasets for for a year and the year's growing season
    (April to october).

    :param input_RET_monthly_dir: Directory file path of downloaded GridMET RET monthly datasets.
    :param output_dir_RET_yearly: File path of directory to save summed GridMET RET data for each year.
    :param output_dir_RET_growing_season: File path of directory to save summed GridMET RET data for each year's
                                     growing season at Western US extent.
    :param year_list: Tuple/list of years_list for which to process data.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_RET_yearly, output_dir_RET_growing_season])

        for year in year_list:  # first loop for years_list
            # # for total years_list
            print(f'summing GridMET RET data for year {year}...')
            openet_datasets = glob(os.path.join(input_RET_monthly_dir, f'*{year}*.tif'))

            # Summing raster for each growing season
            summed_output_for_year = os.path.join(output_dir_RET_yearly, f'GRIDMET_RET_{year}.tif')
            sum_rasters(raster_list=openet_datasets, raster_dir=None, output_raster=summed_output_for_year,
                        ref_raster=openet_datasets[0])

            # # for growing seasons
            print(f'summing GridMET RET data for year {year} growing seasons...')
            openet_datasets = glob(os.path.join(input_RET_monthly_dir, f'*{year}_[4-9]*.tif')) + \
                              glob(os.path.join(input_RET_monthly_dir, f'*{year}_10*.tif'))

            # Summing raster for each growing season
            summed_output_for_grow_season = os.path.join(output_dir_RET_growing_season, f'GRIDMET_RET_{year}.tif')
            sum_rasters(raster_list=openet_datasets, raster_dir=None, output_raster=summed_output_for_grow_season,
                        ref_raster=openet_datasets[0])

    else:
        pass


def convert_prism_data_to_tif(input_dir, output_dir, keyword='prism_tmax'):
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


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, output_dir_prism_yearly=None,
                       year_list=tuple(range(1984, 2021)),
                       keyword='prism_tmax',
                       AZ_shape=AZ_shape,
                       ref_raster=AZ_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and mean to Western US extent) Prism Precipitation, Tmax, and Tmin data. The precipitation data is
    summed for all months in a year.

    :param prism_bil_dir: Directory file path of downloaded prism datasets in .bil format.
    :param prism_tif_dir: Directory file path of prism datasets converted to tif format.
    :param output_dir_prism_monthly: File path of directory to save monthly prism precipitation/temperature data for
                                     at Western US extent.
    :param output_dir_prism_yearly: File path of directory to save summed/mean prism precipitation/temperature data for
                                    each year at Western US extent. Set to None if yearly aggregation is not needed.
    :param year_list: Tuple/list of years_list for which prism data was downloaded.
    :param keyword: keyword to add before processed datasets. Can take 'prism_precip', 'prism_tmax', 'prism_tmin'.
                    Default set to 'prism_tmax'.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: None.
    """

    if not skip_processing:
        if output_dir_prism_yearly is not None:
            makedirs([output_dir_prism_monthly, output_dir_prism_yearly])
        else:
            makedirs([output_dir_prism_monthly])

        convert_prism_data_to_tif(input_dir=prism_bil_dir, output_dir=prism_tif_dir, keyword=keyword)

        #########
        # # Code-block for saving monthly data for the Western US
        #########
        # Clipping Prism monthly datasets for Western US
        monthly_prism_tifs = glob(os.path.join(prism_tif_dir, '*.tif'))  # monthly prism datasets
        for data in monthly_prism_tifs:
            month = os.path.basename(data).split('.')[0][-2:]
            year = os.path.basename(data).split('.')[0].split('_')[2][:4]

            if month.startswith('0'):  # don't want to keep 0 in month for consistency will all datasets
                month = month[-1]

            if 'precip' in keyword:
                monthly_raster_name = f'prism_precip_{year}_{month}.tif'
            elif 'tmax' in keyword:
                monthly_raster_name = f'prism_tmax_{year}_{month}.tif'
            elif 'tmin' in keyword:
                monthly_raster_name = f'prism_tmin_{year}_{month}.tif'

            # the prism datasets are at 4km native resolution and directly clipping and resampling them from 4km
            # resolution creates misalignment of pixels from reference raster. So, first we are resampling CONUS
            # scale original datasets to 2km resolutions and then clipping them at reference raster extent
            clip_resample_reproject_raster(input_raster=data,
                                           input_shape=AZ_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_prism_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=False, ref_raster=ref_raster,
                                           crs='EPSG:26912', resolution=resolution)
    else:
        pass


def sum_cropET_water_yr(years_list, input_cropET_monthly_dir, output_dir_water_yr,
                        save_keyword, skip_processing=False):
    """
    Process (sum) irrigated/rainfed cropET for water year.


    :param years_list: Tuple/list of years_list for which data will be processed.
    :param input_cropET_monthly_dir: Directory file path of downloaded irrigated/rainfed cropET monthly datasets.
    :param output_dir_water_yr: File path of directory to save summed irrigated/rainfed cropET for each water year
                                at Western US extent.
    :param save_keyword: Keyword to use for summed cropET data saving.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_water_yr])

        for yr in years_list:
            print(f'summing monthly cropET for water year {yr}...')

            # summing rainfed/irrigated crop ET for water year (previous year's October to current year's september)
            et_data_prev_years = glob(os.path.join(input_cropET_monthly_dir, f'*{yr - 1}_1[0-2].*tif'))
            et_data_current_years = glob(os.path.join(input_cropET_monthly_dir, f'*{yr}_[1-9].*tif'))
            et_water_yr_list = et_data_prev_years + et_data_current_years

            sum_rasters(raster_list=et_water_yr_list, raster_dir=None,
                        output_raster=os.path.join(output_dir_water_yr, f'{save_keyword}_{yr}.tif'),
                        ref_raster=et_water_yr_list[0])
    else:
        pass


def create_slope_raster(input_raster, output_dir, raster_name, skip_processing=False):
    """
    Create Slope raster in Percent from DEM raster.

    :param input_raster: Input raster filepath.
    :param output_dir: Output raster directory filepath.
    :param raster_name: Output raster name.
    :param skip_processing: Set to True if want to skip slope processing.

    :return: None.
    """
    if not skip_processing:
        dem_options = gdal.DEMProcessingOptions(format="GTiff", computeEdges=True, alg='ZevenbergenThorne',
                                                slopeFormat='percent', scale=100000)

        makedirs([output_dir])
        output_raster = os.path.join(output_dir, raster_name)

        slope_raster = gdal.DEMProcessing(destName=output_raster, srcDS=input_raster, processing='slope',
                                          options=dem_options)

        del slope_raster
    else:
        pass


def process_AWC_data(input_dir, shape, output_dir, ref_raster=AZ_raster,
                     resolution=model_res, skip_processing=False):
    """
    Process available water capacity (AWC) data for Western US

    :param input_dir: Filepath of input directory.
    :param shape: Filepath of shapefile to be used for clipping.
    :param output_dir: Filepath of output directory.
    :param ref_raster: Fileapth of Western US reference raster.
    :param resolution: Model resolution.
    :param skip_processing: Set to True to skip the process.

    :return: None
    """
    if not skip_processing:
        makedirs([output_dir])

        AWC_raster = glob(os.path.join(input_dir, '*.tif'))[0]

        clip_resample_reproject_raster(input_raster=AWC_raster, input_shape=shape, output_raster_dir=output_dir,
                                       raster_name='AWC.tif', clip_and_resample=True, resolution=resolution,
                                       crs='EPSG:26912', ref_raster=ref_raster)


def extract_month_from_GrowSeason_data(GS_data_dir, skip_processing=False):
    """
    Extract start and ending growing season months from growing season dataset (provided by Justin Huntington DRI;
    downloaded from GEE to google drive). The output datasets have 2 bands, containing start and end month info,
    respectively.

    :param GS_data_dir: Directory path of growing season dataset. The GEE-downloaded datasets are in the
                        'ee_exports' folder.
    :param skip_processing: Set to true if want to skip processing.

    :return: None.
    """

    def doy_to_month(year, doy):
        """
        Convert a day of year (DOY) to a month in a given year.

        :return: Month of the corresponding date.
        """
        if np.isnan(doy):  # Check if the DOY is NaN
            return np.nan

        # January 1st of the given year + timedelta of the DoY to extract month
        month = (datetime.datetime(year, 1, 1) + datetime.timedelta(int(doy) - 1)).month

        return month

    if not skip_processing:
        print('Processing growing season data...')

        # collecting GEE exported data files and making new directories for processing
        GS_data_files = glob(os.path.join(GS_data_dir, 'ee_exports', '*.tif'))
        interim_dir = os.path.join(GS_data_dir, 'interim')
        makedirs([interim_dir])

        # looping through each dataset, extracting start and end of the growing season months, saving as an array
        for data in GS_data_files:
            raster_name = os.path.basename(data)
            year = int(raster_name.split('_')[1].split('.')[0])

            # clipping and resampling the growing season data with the Arizona reference raster
            interim_raster = clip_resample_reproject_raster(input_raster=data,
                                                            input_shape=AZ_shape,
                                                            raster_name=raster_name,
                                                            output_raster_dir=interim_dir,
                                                            clip=False, resample=False, clip_and_resample=True,
                                                            targetaligned=True, resample_algorithm='near',
                                                            use_ref_width_height=False, ref_raster=None,
                                                            crs='EPSG:26912',
                                                            resolution=model_res)

            # reading the start and end DoY of the growing season
            startDOY_arr, ras_file = read_raster_arr_object(interim_raster, band=1)
            endDOY_arr = read_raster_arr_object(interim_raster, band=2, get_file=False)

            # vectorizing the doy_to_month() function to apply on a numpy array
            vectorized_doy_to_date = np.vectorize(doy_to_month)

            # converting the start and end DoY to corresponding month
            start_months = vectorized_doy_to_date(year, startDOY_arr)
            end_months = vectorized_doy_to_date(year, endDOY_arr)

            # stacking the arrays together (single tif with 2 bands)
            GS_month_arr = np.stack((start_months, end_months), axis=0)

            # saving the array
            output_raster = os.path.join(GS_data_dir, raster_name)
            with rio.open(
                    output_raster,
                    'w',
                    driver='GTiff',
                    height=GS_month_arr.shape[1],
                    width=GS_month_arr.shape[2],
                    dtype=np.float32,
                    count=GS_month_arr.shape[0],
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(GS_month_arr)


def dynamic_gs_sum_ET(year_list, growing_season_dir, monthly_input_dir,
                      gs_output_dir, sum_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) sums effective precipitation and irrigated crop ET monthly rasters for
    the growing seasons.

    :param year_list: List of years_list to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly effective precipitation/irrigated crop ET datasets.
    :param gs_output_dir:  Directory path (output) for summed growing season effective precipitation/irrigated crop ET
                           datasets.
    :param sum_keyword: Keyword str to add before the summed raster.
                       Should be 'effective_precip' or 'Irrigated_cropET' or 'OpenET_ensemble'
    :param skip_processing: Set to True if want to skip processing this step.

    :return:
    """
    if not skip_processing:
        print(f'Dynamically summing {sum_keyword} monthly datasets for growing season...')

        makedirs([gs_output_dir])

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:

            # gathering and sorting the peff datasets by month (from 1 to 12)
            datasets = glob(os.path.join(monthly_input_dir, f'*{year}*.tif'))
            sorted_datasets = sorted(datasets, key=lambda x: int(
                month_pattern.search(x).group(1)))  # First capturing group (the month)

            # peff/cropET monthly array stacked in a single numpy array
            arrs_stck = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_datasets], axis=0)

            # gathering, reading, and stacking growing season array
            gs_data = glob(os.path.join(growing_season_dir, f'*{year}*.tif'))[0]
            start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)  # band 1
            end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)  # band 2

            # We create a 1 pixel "kernel", representing months 1 to 12 (shape : 12, 1, 1).
            # Then it is broadcasted across the array and named as the kernel_mask.
            # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
            kernel = np.arange(1, 13, 1).reshape(12, 1, 1)
            kernel_mask = (kernel >= start_gs_arr) & (kernel <= end_gs_arr)

            # sum peff/cropET arrays over the valid months using the kernel_mask
            summed_arr = np.sum(arrs_stck * kernel_mask, axis=0)

            # saving the summed peff array
            output_name = f'{sum_keyword}_{year}.tif'
            output_path = os.path.join(gs_output_dir, output_name)
            with rio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=summed_arr.shape[0],
                    width=summed_arr.shape[1],
                    dtype=np.float32,
                    count=1,
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(summed_arr, 1)


def dynamic_gs_sum_peff_with_3m_SM_storage(year_list, growing_season_dir, monthly_input_dir,
                                           gs_output_dir, skip_processing=False):
    """
    Dynamically (spatio-temporally) sums effective precipitation (peff) monthly rasters for
    the growing seasons with 3 month's lag peff included before the gorwing season starts.

    :param year_list: List of years_list to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly effective precipitation/irrigated crop ET datasets.
    :param gs_output_dir:  Directory path (output) for summed growing season effective precipitation/irrigated crop ET
                           datasets.
    :param skip_processing: Set to True if want to skip processing this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([gs_output_dir])

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern_current_yr = re.compile(r'_([0-9]{1,2})\.tif')
        month_pattern_prev_yr = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            print(f'Dynamically summing effective precipitation monthly datasets for growing season {year}...')

            # current year: gathering and sorting the peff datasets by month for current year (from 1 to 12)
            datasets_current_yr = glob(os.path.join(monthly_input_dir, f'*{year}*.tif'))
            sorted_datasets_current_yr = sorted(datasets_current_yr, key=lambda x: int(
                month_pattern_current_yr.search(x).group(1)))  # First capturing group (the month)

            # current year: peff monthly array stacked in a single numpy array
            arrs_stck_current_yr = np.stack(
                [read_raster_arr_object(i, get_file=False) for i in sorted_datasets_current_yr], axis=0)

            # previous year: gathering datasets for months 10-12 of the previous year
            datasets_previous_year = glob(os.path.join(monthly_input_dir, f'*{year - 1}*.tif'))
            datasets_prev_10_12 = [f for f in datasets_previous_year if
                                   10 <= int(month_pattern_prev_yr.search(f).group(1)) <= 12]
            sorted_datasets_prev_yr = sorted(datasets_prev_10_12,
                                             key=lambda x: int(month_pattern_prev_yr.search(x).group(1)))

            # previous year: peff monthly array stacked in a single numpy array
            arrs_stck_prev_yr = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_datasets_prev_yr],
                                         axis=0)

            # gathering, reading, and stacking growing season array
            gs_data = glob(os.path.join(growing_season_dir, f'*{year}*.tif'))[0]
            start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)  # band 1
            end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)  # band 2

            # current year: deduct 3 months from start_gs_arr to consider the effect of  3 months' peff storage
            # then finalize the current year's start season array
            # (where the value <= 0, set to 1 otherwise it's already been adjusted by deducting 3)
            start_gs_arr_adjusted = start_gs_arr - 3
            start_gs_arr_current_yr = np.where(start_gs_arr_adjusted <= 0, 1, start_gs_arr_adjusted)
            end_gs_arr_current_yr = end_gs_arr

            # previous year: create start and end season array
            start_gs_arr_prev_yr = np.where(start_gs_arr_adjusted <= 0, start_gs_arr_adjusted + 12, np.nan)
            end_gs_arr_prev_yr = np.where(start_gs_arr_adjusted <= 0, 12, np.nan)

            # ****** Summing peff for the current year (using start_gs_arr_current_yr and end_gs_arr_current_yr) ******

            # We create a 1 pixel "kernel", representing months 1 to 12 (shape : 12, 1, 1).
            # Then it is broadcasted across the array and named as the kernel_mask.
            # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
            kernel_current_year = np.arange(1, 13, 1).reshape(12, 1, 1)
            kernel_mask_current_year = (kernel_current_year >= start_gs_arr_current_yr) & (
                    kernel_current_year <= end_gs_arr_current_yr)

            # sum peff arrays over the valid months using the kernel_mask
            summed_arr_current_yr = np.sum(arrs_stck_current_yr * kernel_mask_current_year, axis=0)

            # ****** Summing peff for the previous year (using start_gs_arr_prev_yr and end_gs_arr_prev_yr) ******

            # We create a 1 pixel "kernel", representing months 10 to 12 (shape : 3, 1, 1).
            # Then it is broadcasted across the array and named as the kernel_mask.
            # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
            kernel_prev_year = np.arange(10, 13, 1).reshape(3, 1, 1)
            kernel_mask_prev_year = (kernel_prev_year >= start_gs_arr_prev_yr) & (
                    kernel_prev_year <= end_gs_arr_prev_yr)

            # sum peff arrays over the valid months using the kernel_mask
            summed_arr_prev_yr = np.sum(arrs_stck_prev_yr * kernel_mask_prev_year, axis=0)

            # ****** Combine the results from the current year and previous year ******
            summed_total_arr = np.sum([summed_arr_current_yr, summed_arr_prev_yr], axis=0)

            # saving the summed peff array
            output_name = f'effective_precip_{year}.tif'
            output_path = os.path.join(gs_output_dir, output_name)
            with rio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=summed_total_arr.shape[0],
                    width=summed_total_arr.shape[1],
                    dtype=np.float32,
                    count=1,
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(summed_total_arr, 1)


def accumulate_monthly_datasets_to_water_year(skip_processing=False):
    """
    accumulates monthly datasets to water year by sum or mean.

    :param skip_processing: Set to true to skip this processing step.

    :return: False.
    """
    monthly_data_path_dict = {
        'GRIDMET_Precip': '../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_monthly',
        'PRISM_Tmax': '../../Data_main/AZ_files/rasters/PRISM_Tmax/WestUS_monthly',
        'GRIDMET_RET': '../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_monthly',
        'GRIDMET_max_RH': '../../Data_main/AZ_files/rasters/GRIDMET_max_RH/WestUS_monthly',
        'GRIDMET_short_rad': '../../Data_main/AZ_files/rasters/GRIDMET_short_rad/WestUS_monthly',
        'DAYMET_sun_hr': '../../Data_main/AZ_files/rasters/DAYMET_sun_hr/WestUS_monthly',
        'Rainy_days': '../../Data_main/AZ_files/rasters/Rainy_days/WestUS_monthly'}

    water_yr_accum_dict = {
        'Irrigated_cropET': 'sum',
        'PRISM_Tmax': 'mean',
        'GRIDMET_Precip': ['sum', 'mean'],
        'GRIDMET_RET': 'sum',
        'GRIDMET_max_RH': 'mean',
        'GRIDMET_short_rad': 'mean',
        'DAYMET_sun_hr': 'mean',
        'Rainy_days': 'sum'}

    if not skip_processing:
        # # processing the variables/predictors
        for var, path in monthly_data_path_dict.items():
            print(f'accumulating monthly datasets to water year for {var}...')

            if var == 'Irrigated_cropET':
                output_dir = os.path.join(os.path.dirname(path), 'WestUS_water_year')
                years_to_run = range(1986, 2024 + 1)

            else:
                output_dir = os.path.join(os.path.dirname(path), 'WestUS_water_year')
                years_to_run = range(1985, 2024 + 1)

            makedirs([output_dir])

            # accumulate by sum or mean
            accum_by = water_yr_accum_dict[var]

            for yr in years_to_run:
                # collecting monthly datasets for the water year
                data_prev_years = glob(os.path.join(path, f'*{yr - 1}_1[0-2].*tif'))
                data_current_years = glob(os.path.join(path, f'*{yr}_[1-9].*tif'))
                total_data_list = data_prev_years + data_current_years

                # data name extraction
                data_name_extraction = os.path.basename(total_data_list[0]).split('_')[:-2]
                data_name = '_'.join(data_name_extraction) + f'_{yr}' + '.tif'

                # sum() or mean() accumulation
                if var in ['GRIDMET_Precip', 'TERRACLIMATE_SR']:  # we perform both mean and sum
                    sum_rasters(raster_dir=None, raster_list=total_data_list,
                                output_raster=os.path.join(output_dir, 'sum', data_name),
                                ref_raster=total_data_list[0], nodata=no_data_value)

                    mean_rasters(raster_dir=None, raster_list=total_data_list,
                                 output_raster=os.path.join(output_dir, 'mean', data_name),
                                 ref_raster=total_data_list[0], nodata=no_data_value)

                else:
                    if accum_by == 'sum':
                        sum_rasters(raster_dir=None, raster_list=total_data_list,
                                    output_raster=os.path.join(output_dir, data_name),
                                    ref_raster=total_data_list[0], nodata=no_data_value)
                    elif accum_by == 'mean':
                        mean_rasters(raster_dir=None, raster_list=total_data_list,
                                     output_raster=os.path.join(output_dir, data_name),
                                     ref_raster=total_data_list[0], nodata=no_data_value)
    else:
        pass


def estimate_precip_intensity_water_yr(years_list, input_dir_precip, input_dir_rainy_day, output_dir,
                                       nodata=no_data_value, skip_processing=False):
    """
    Estimate precipitation intensity (water year precipitation / num of rainy days).

    :param years_list: List of years to process data for.
    :param input_dir_precip: Filepath of water year summed precipitation data directory.
    :param input_dir_rainy_day: Filepath of water year summed rainy days data directory.
    :param output_dir: Filepath of output directory.
    :param nodata: No data value. Default set to -9999.
    :param skip_processing: Set to true if want to skip this process.

    :return: None
    """
    if not skip_processing:
        makedirs([output_dir])

        for year in years_list:
            print(f'estimating water year precipitation intensity for year {year}...')

            # loading and reading datasets
            precip_data = glob(os.path.join(input_dir_precip, f'*{year}*.tif'))[0]
            rainy_data = glob(os.path.join(input_dir_rainy_day, f'*{year}*.tif'))[0]

            precip_arr, raster_file = read_raster_arr_object(precip_data)
            rainy_arr = read_raster_arr_object(rainy_data, get_file=False)

            # calculating precipitation intensity (water year precipitation / num of rainy days)
            intensity_arr = np.where((precip_arr != -9999) & (rainy_arr != -9999) & (rainy_arr != 0),
                                     precip_arr / rainy_arr, nodata)

            # saving estimated raster
            output_raster = os.path.join(output_dir, f'Precipitation_intensity_{year}.tif')
            write_array_to_raster(intensity_arr, raster_file, raster_file.transform, output_raster)

    else:
        pass


def estimate_PET_by_P_water_yr(years_list, input_dir_PET, input_dir_precip, output_dir,
                               nodata=no_data_value, skip_processing=False):
    """
    Estimate PET/P (dryness index) for water year.

    :param years_list: List of years to process data for.
    :param input_dir_PET: ilepath of water year summed PET data directory.
    :param input_dir_precip: Filepath of water year summed precipitation data directory.
    :param output_dir: Filepath of output directory.
    :param nodata: No data value. Default set to -9999.
    :param skip_processing: Set to True if want to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        for year in years_list:
            print(f'estimating water year PET/P for year {year}...')

            # loading and reading datasets
            pet_data = glob(os.path.join(input_dir_PET, f'*{year}*.tif'))[0]
            precip_data = glob(os.path.join(input_dir_precip, f'*{year}*.tif'))[0]

            pet_arr = read_raster_arr_object(pet_data, get_file=False)
            precip_arr, raster_file = read_raster_arr_object(precip_data)

            # calculating PET/P
            dry_arr = np.where((precip_arr != -9999) & (pet_arr != -9999), pet_arr / precip_arr, nodata)

            # saving estimated raster
            output_raster = os.path.join(output_dir, f'dryness_index_{year}.tif')
            write_array_to_raster(dry_arr, raster_file, raster_file.transform, output_raster)

    else:
        pass


def develop_P_PET_correlation_dataset(monthly_precip_dir, monthly_pet_dir,
                                      output_dir, skip_processing=False):
    """
    Develop PET and P correaltion dataset (static) for the Western US.

    :param monthly_precip_dir: Filepath of monthly precip directory.
    :param monthly_pet_dir: Filepath of monthly pet directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip creating this dataset.

    :return: None
    """
    if not skip_processing:
        print('creating P-PET correlation dataset...')

        makedirs([output_dir])

        # accumulating precip and pet data
        monthly_precip_data_list = glob(os.path.join(monthly_precip_dir, '*.tif'))
        monthly_pet_data_list = glob(os.path.join(monthly_pet_dir, '*.tif'))

        # reading datasets as arrays
        monthly_precip_arr_list = [read_raster_arr_object(i, get_file=False) for i in monthly_precip_data_list]
        monthly_pet_arr_list = [read_raster_arr_object(i, get_file=False) for i in monthly_pet_data_list]

        # stacking monthly datasets into a list
        precip_stack = np.stack(monthly_precip_arr_list, axis=0)  # shape becomes - n_months, n_lat (height), n_lon(width)
        pet_stack = np.stack(monthly_pet_arr_list, axis=0)  # shape becomes - n_months, n_lat (height), n_lon(width)

        # Calculating mean along the time axis (i.e., across months) for each pixel
        precip_mean = np.mean(precip_stack, axis=0)
        pet_mean = np.mean(pet_stack, axis=0)

        # estimating precip and pet anomalies
        precip_anomalies = precip_stack - precip_mean
        pet_anomalies = pet_stack - pet_mean

        # getting numerator (covariance) for each pixel across time
        numerator = np.sum(precip_anomalies * pet_anomalies, axis=0)

        # getting denominator (sum of squares for both variables (this measures the total variation for each))
        sum_of_squares_precip = np.sqrt(np.sum(precip_anomalies ** 2, axis=0))
        sum_of_squares_pet = np.sqrt(np.sum(pet_anomalies ** 2, axis=0))
        denominator = sum_of_squares_precip * sum_of_squares_pet

        # calculating Pearson correlation for each pixel
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_arr = numerator / denominator

        output_raster = os.path.join(output_dir, 'PET_P_corr.tif')
        _, ref_file = read_raster_arr_object(monthly_precip_data_list[0])
        write_array_to_raster(correlation_arr, ref_file, ref_file.transform, output_raster)

    else:
        pass


def process_lake_raster(lake_raster, AZ_shape, output_dir, skip_processing=False):
    """
    Mask lake raster for Arizona.

    :param lake_raster: Filepatth of lake raster for WestUS.
    :param AZ_shape: Arizona shapefile.
    :param output_dir: Output directory to save the output raster.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        clip_resample_reproject_raster(input_raster=lake_raster, input_shape=AZ_shape,
                                       output_raster_dir=output_dir,
                                       raster_name='Lakes_AZ.tif',
                                       clip_and_resample=True,
                                       resolution=model_res,
                                       crs='EPSG:26912',
                                       ref_raster=AZ_raster)
    else:
        pass


def interpolate_missing_Daymet_sunHr_data(years_to_interpolate, daymet_data_dir, skip_processing=False):
    """
    interpolate missing monthly data for daymet sun dr.


    :param years_to_interpolate: List of year to interpolate for.
    :param daymet_data_dir: Daymet monthly data directory.
    :param skip_processing: Set to true to skip this process.

    :return: None.
    """
    if not skip_processing:
        print('interpolating Daymet Sun Hr data...')

        for year in years_to_interpolate:
            for month in range(1, 13):
                if year == 2023 and month in range(1, 12):          # 2023 has data available from month 1-11
                    continue
                else:
                    # reading previous two years' data
                    data_two_year_back = glob(os.path.join(daymet_data_dir, f'*{year-2}_{month}.tif'))[0]
                    data_one_year_back = glob(os.path.join(daymet_data_dir, f'*{year-1}_{month}.tif'))[0]

                    arr_two, file = read_raster_arr_object(data_two_year_back)
                    arr_one = read_raster_arr_object(data_one_year_back, get_file=False)

                    # averaging last two years' data to interpolate missing data
                    stacked_arr = np.stack([arr_two, arr_one], axis=0)
                    new_arr = np.nanmean(stacked_arr, axis=0)

                    # saving
                    output_path = os.path.join(daymet_data_dir, f'DAYMET_sun_hr_{year}_{month}.tif')
                    write_array_to_raster(new_arr, file, file.transform, output_path)


def run_all_preprocessing(skip_process_GrowSeason_data=False,
                          skip_merging_irrigated_frac=False,
                          skip_merging_irrigated_cropET=False,
                          skip_classifying_irrigated_cropland=False,
                          skip_filtering_irrigated_cropET=False,
                          skip_gridmet_precip_processing=False,
                          skip_gridmet_RET_precessing=False,
                          skip_summing_irrigated_cropET_gs=False,
                          skip_prism_processing=False,
                          skip_processing_slope_data=False,
                          skip_process_AWC_data=False,
                          skip_interpolate_daymet = False,
                          skip_accum_to_water_year_datasets=False,
                          skip_summing_irrigated_cropET_water_yr=False,
                          skip_estimate_precip_intensity=False,
                          skip_estimate_dryness_index=False,
                          skip_create_P_PET_corr_dataset=False,
                          skip_process_lake_raster=False,
                          ref_raster=AZ_raster):
    """
    Run all preprocessing steps.

    :param skip_process_GrowSeason_data: Set to True to skip processing growing season data.
    :param skip_merging_irrigated_frac: Set to True to skip merging irrigated fraction data.
    :param skip_merging_irrigated_cropET: Set to True to skip merging irrigated cropET data.
    :param skip_classifying_irrigated_cropland: Set to True if want to skip classifying irrigated  cropland data.
    :param skip_filtering_irrigated_cropET: Set to True if want to skip filtering irrigated cropET data.
    :param skip_prism_processing: Set to True to skip prism tmax processing.
    :param skip_gridmet_precip_processing: Set True to skip gridmet precip yearly data processing.
    :param skip_gridmet_RET_precessing: Set to True to skip GridMET RET data processing.
    :param skip_summing_irrigated_cropET_gs: Set to True if want to skip summing irrigated cropET data summing for year/grow season.
    :param skip_processing_slope_data: Set to True if want to skip DEM to slope conversion.
    :param skip_process_AWC_data: Set to True ti skip processing AWC data.
    :param skip_interpolate_daymet: Sett to true to skip interpoalte Daymet Sun Hr data.
    :param skip_accum_to_water_year_datasets: Set to True to skip accumulating monthly dataset to water year.
    :param skip_summing_irrigated_cropET_water_yr: Set to True if want to skip summing irrigated cropET for water year.
    :param skip_estimate_precip_intensity: Set to True to skip processing water year precipitation intensity data.
    :param skip_estimate_dryness_index: Set to True to skip processing water year PET/P (dryness Index) data.
    :param skip_create_P_PET_corr_dataset: Set to True to skip create P-PET correlation dataset.
    :param skip_process_lake_raster: Set to True to process lake raster.
    :param ref_raster: Filepath of Western US reference raster to use in 2km pixel lat-lon raster creation and to use
                       as reference raster in other processing operations.

    :return: None.
    """
    # process growing season data
    extract_month_from_GrowSeason_data(GS_data_dir='../../Data_main/AZ_files/rasters/Growing_season',
                                       skip_processing=skip_process_GrowSeason_data)

    # merge irrigated fraction dataset
    merge_GEE_data_patches_IrrMapper_extents(year_list=list(range(1985, 2025)),
                                             input_dir_irrmapper='../../Data_main/AZ_files/rasters/Irrigation_Frac_IrrMapper',
                                             merged_output_dir='../../Data_main/AZ_files/rasters/Irrigated_cropland/Irrigated_Frac',
                                             merge_keyword='Irrigated_Frac', monthly_data=False,
                                             ref_raster=GEE_merge_ref_raster_30m,
                                             skip_processing=skip_merging_irrigated_frac)

    # merge irrigated cropET dataset
    merge_GEE_data_patches_IrrMapper_extents(year_list=list(range(1985, 2025)),
                                             input_dir_irrmapper='../../Data_main/AZ_files/rasters/Irrig_crop_OpenET_IrrMapper',
                                             merged_output_dir='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_monthly_raw',
                                             merge_keyword='Irrigated_cropET', monthly_data=True,
                                             ref_raster=GEE_merge_ref_raster_30m,
                                             skip_processing=skip_merging_irrigated_cropET)

    # classify irrigated cropland data
    classify_irrigated_cropland(
        irrigated_fraction_dir='../../Data_main/AZ_files/rasters/Irrigated_cropland/Irrigated_Frac',
        irrigated_cropland_output_dir='../../Data_main/AZ_files/rasters/Irrigated_cropland',
        skip_processing=skip_classifying_irrigated_cropland)

    # filtering irrigated cropET with irrigated cropland data
    filter_irrigated_cropET_with_irrigated_cropland(
        irrigated_cropland_dir='../../Data_main/AZ_files/rasters/Irrigated_cropland',
        irrigated_cropET_input_dir='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_monthly_raw',
        irrigated_cropET_output_dir='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_monthly',
        skip_processing=skip_filtering_irrigated_cropET)

    # sum monthly irrigated cropET for dynamic growing season
    dynamic_gs_sum_ET(year_list=list(range(1985, 2025)),
                      growing_season_dir='../../Data_main/AZ_files/rasters/Growing_season',
                      monthly_input_dir='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_monthly',
                      gs_output_dir='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_grow_season',
                      sum_keyword='Irrigated_cropET',
                      skip_processing=skip_summing_irrigated_cropET_gs)

    # prism maximum temperature data processing
    process_prism_data(year_list=tuple(range(1984, 2025)),
                       prism_bil_dir='../../Data_main/AZ_files/rasters/PRISM_Tmax/bil_format',
                       prism_tif_dir='../../Data_main/AZ_files/rasters/PRISM_Tmax/tif_format',
                       output_dir_prism_monthly='../../Data_main/AZ_files/rasters/PRISM_Tmax/WestUS_monthly',
                       output_dir_prism_yearly=None,
                       AZ_shape=AZ_shape,
                       keyword='prism_tmax', skip_processing=skip_prism_processing)

    # gridmet precip yearly data processing
    sum_GridMET_precip_yearly_data(
        year_list=list(range(1984, 2025)),
        input_gridmet_monthly_dir='../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_monthly',
        output_dir_yearly='../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_yearly',
        skip_processing=skip_gridmet_precip_processing)


    # GridMET yearly data processing
    sum_GridMET_RET_yearly_data(year_list=list(range(1984, 2025)),
                                input_RET_monthly_dir='../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_monthly',
                                output_dir_RET_yearly='../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_yearly',
                                output_dir_RET_growing_season='../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_grow_season',
                                skip_processing=skip_gridmet_RET_precessing)

    # converting DEM data to slope
    create_slope_raster(input_raster='../../Data_main/AZ_files/rasters/DEM/WestUS/DEM.tif',
                        output_dir='../../Data_main/AZ_files/rasters/Slope/WestUS', raster_name='Slope.tif',
                        skip_processing=skip_processing_slope_data)

    # processing available water capacity (AWC) data
    process_AWC_data(input_dir='../../Data_main/AZ_files/rasters/Available_water_capacity/awc_gNATSGO',
                     shape=AZ_shape,
                     output_dir='../../Data_main/AZ_files/rasters/Available_water_capacity/WestUS',
                     ref_raster=ref_raster, resolution=model_res,
                     skip_processing=skip_process_AWC_data)

    # interpolating Daymet un hr data
    interpolate_missing_Daymet_sunHr_data(years_to_interpolate=[2023, 2024],
                                          daymet_data_dir='../../Data_main/AZ_files/rasters/DAYMET_sun_hr/WestUS_monthly',
                                          skip_processing=skip_interpolate_daymet)

    # # # # # # # # # # # # # # # # # # # # # # for water year model # # # # # # # # # # # # # # # # # # # # # # # # # #

    # accumulating monthly dataset to water year
    accumulate_monthly_datasets_to_water_year(skip_processing=skip_accum_to_water_year_datasets)

    # sum monthly irrigated cropET for water year
    sum_cropET_water_yr(years_list=list(range(1986, 2025)),
                        input_cropET_monthly_dir='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_monthly',
                        output_dir_water_yr='../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_water_year',
                        save_keyword='Irrigated_cropET',
                        skip_processing=skip_summing_irrigated_cropET_water_yr)


    # estimate water year precipitation intensity (precipitation / rainy days)
    estimate_precip_intensity_water_yr(years_list=list(range(1985, 2025)),
                                       input_dir_precip='../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_water_year/mean',
                                       input_dir_rainy_day='../../Data_main/AZ_files/rasters/Rainy_days/WestUS_water_year',
                                       output_dir='../../Data_main/AZ_files/rasters/Precipitation_intensity',
                                       skip_processing=skip_estimate_precip_intensity)

    # estimate PET/P (dryness index) for water year
    estimate_PET_by_P_water_yr(years_list=list(range(1985, 2025)),
                               input_dir_PET='../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_water_year',
                               input_dir_precip='../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_water_year/sum',
                               output_dir='../../Data_main/AZ_files/rasters/Dryness_index',
                               skip_processing=skip_estimate_dryness_index)

    # create P-PET correlation dataset
    develop_P_PET_correlation_dataset(monthly_precip_dir='../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_monthly',
                                      monthly_pet_dir='../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_monthly',
                                      output_dir='../../Data_main/AZ_files/rasters/P_PET_correlation',
                                      skip_processing=skip_create_P_PET_corr_dataset)

    # process (mask) lake raster
    process_lake_raster(lake_raster='../../Data_main/AZ_files/rasters/HydroLakes/Lakes.tif',
                        AZ_shape=AZ_shape, output_dir='../../Data_main/AZ_files/rasters/HydroLakes',
                        skip_processing=skip_process_lake_raster)