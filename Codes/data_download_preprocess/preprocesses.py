import os
import re
import sys
import shutil
import datetime
import numpy as np
from glob import glob
from osgeo import gdal
import rasterio as rio
import geopandas as gpd

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, mosaic_rasters_list, \
    clip_resample_reproject_raster, sum_rasters, mean_rasters, make_lat_lon_array_from_raster
from Codes.effective_precip.m00_eff_precip_utils import estimate_peff_precip_water_year_fraction

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def apply_maximum_occurrence_approach(input_rasters_list, output_dir, raster_name):
    """
    **** Best works for binary rasters ****
    Creates a output binary raster using maximum occurrence approach (a pixel is 1 if most of the input rasters
    show 1 in that pixel) incorporating the input rasters.

    :param input_rasters_list: A list of input rasters.
    :param output_dir: Filepath of output directory.
    :param raster_name: Output raster name. Must include '.tif'.

    :return: The output raster.
    """
    # Making one raster as the reference raster
    ref_arr, ref_file = read_raster_arr_object(input_rasters_list[0], change_dtype=False)
    ref_shape = ref_arr.shape

    # making input dataset placeholder names
    input_raster_names = [f'input_raster_{i}' for i in range(1, len(input_rasters_list) + 1)]

    # making a dictionary where each raster array (after data reading) will be assigned to individual names
    input_raster_dict = {}
    for name, data in zip(input_raster_names, input_rasters_list):
        arr = read_raster_arr_object(data, get_file=False, change_dtype=False).flatten()
        # the max occurrence approach uses np.bincount() which can't take non-negative element. So, replacing all
        # nan/no data (-9999) values with 0
        arr[arr == -9999] = 0

        # adding the array to the dictionary
        input_raster_dict[name] = arr

    # stacking the data arrays
    array_list = [arr for arr in input_raster_dict.values()]
    max_arr = np.stack(array_list, axis=0)

    # creating an empty array to assign cropland (1) if most years_list between 2008-2011 are cropland
    new_arr = np.zeros(array_list[0].shape, dtype=np.int8)  # this is a flatten shape array

    # this loop takes each column of stacked array. np.bincount() counts the number of occurrence of each value
    # in the selected column. Then, np.argmax() selects the value which occurs most frequently. Then, we paste
    # the selected value to a new array

    for i in range(max_arr.shape[1]):
        selected_arr = max_arr[:, i]
        selected_arr = np.where(np.isnan(selected_arr), 0, selected_arr)  # replacing Nan values with 0
        max_occurred_value = np.argmax(np.bincount(list(selected_arr)))  # # # improve this. try to avoid loop
        new_arr[i] = max_occurred_value

    max_arr = new_arr.reshape(ref_shape)

    # converting the data type to Float32
    max_arr = max_arr.astype(np.float32)

    # replacing the nan pixels with -9999 value
    max_arr = np.where(~np.isnan(ref_arr), max_arr, ref_arr)

    # Writing result of maximum occurrence
    makedirs([output_dir])
    output_raster = os.path.join(output_dir, raster_name)
    write_array_to_raster(raster_arr=max_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)

    return output_raster


def merge_GEE_data_patches_IrrMapper_LANID_extents(year_list, input_dir_irrmapper, input_dir_lanid, merged_output_dir,
                                                   merge_keyword, monthly_data=True, ref_raster=WestUS_raster,
                                                   skip_processing=False):
    """
    Merge/mosaic downloaded GEE data for IrrMapper and LANID extent.

    :param year_list: Tuple/list of years_list for which data will be processed.
    :param input_dir_irrmapper: Input directory filepath of datasets at IrrMapper extent.
    :param input_dir_lanid: Input directory filepath of datasets at LANID extent.
    :param merged_output_dir: Output directory filepath to save merged data.
    :param merge_keyword: Keyword to use while merging. Foe example: 'Rainfed_Frac', 'Irrigated_crop_OpenET', etc.
    :param monthly_data: Boolean. If False will look/search for yearly data patches. Default set to True to look for
                         monthly datasets.
    :param ref_raster: Reference raster to use in merging. Default set to Western US reference raster.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    :return: None.
    """
    if not skip_processing:
        makedirs([merged_output_dir])

        if monthly_data:  # for datasets that are monthly
            month_list = list(range(1, 13))

            for year in year_list:
                for month in month_list:
                    search_by = f'*{year}_{month}_*.tif'

                    # making input raster list by joining rasters of irrmapper extent and rasters of lanid extent
                    irrmapper_raster_list = glob(os.path.join(input_dir_irrmapper, search_by))
                    lanid_raster_list = glob(os.path.join(input_dir_lanid, search_by))
                    irrmapper_raster_list.extend(lanid_raster_list)

                    total_raster_list = irrmapper_raster_list

                    if len(total_raster_list) > 0:  # to only merge for years_list and months when data is available
                        merged_raster_name = f'{merge_keyword}_{year}_{month}.tif'
                        mosaic_rasters_list(input_raster_list=total_raster_list, output_dir=merged_output_dir,
                                            raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                            resampling_method='nearest', mosaicing_method='first',
                                            resolution=model_res, nodata=no_data_value)

                        print(f'{merge_keyword} data merged for year {year}, month {month}')

        else:  # for datasets that are yearly
            for year in year_list:
                search_by = f'*{year}_*.tif'

                # making input raster list by joining rasters of irrmapper extent and rasters of lanid extent
                irrmapper_raster_list = glob(os.path.join(input_dir_irrmapper, search_by))
                lanid_raster_list = glob(os.path.join(input_dir_lanid, search_by))

                irrmapper_raster_list.extend(lanid_raster_list)

                total_raster_list = irrmapper_raster_list

                if len(total_raster_list) > 0:  # to only merge for years_list and months when data is available
                    merged_raster_name = f'{merge_keyword}_{year}.tif'
                    mosaic_rasters_list(input_raster_list=total_raster_list, output_dir=merged_output_dir,
                                        raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                        resampling_method='nearest', mosaicing_method='first',
                                        resolution=model_res, nodata=no_data_value)

                    print(f'{merge_keyword} data merged for year {year}')
    else:
        pass


def classify_irrigated_rainfed_cropland(rainfed_fraction_dir, irrigated_fraction_dir, tree_cover_dir,
                                        rainfed_cropland_output_dir, irrigated_cropland_output_dir,
                                        skip_processing=False):
    """
    Classifies rainfed and irrigated cropland using rainfed and irrigated fraction data.

    ** The rainfed fraction data is only available for 2008-2020, as CDL has Western US scale crop classification data
    starting from 2008. This function classifies rainfed cropland data for 2008-2020 only, but irrigated cropland data
    for 2000-2020. **

    :param rainfed_fraction_dir: Input directory path for rainfed fraction data.
    :param irrigated_fraction_dir: Input directory path for irrigated fraction data.
    :param tree_cover_dir: Input directory for tree cover dataset.
    :param rainfed_cropland_output_dir: Output directory path for classified rainfed cropland data.
    :param irrigated_cropland_output_dir: Output directory path for classified irrigated cropland data.
    :param skip_processing: Set to True if want to skip classifying irrigated and rainfed cropland data.

    :return: None
    """
    if not skip_processing:
        makedirs([rainfed_cropland_output_dir, irrigated_cropland_output_dir])

        ############################
        # # Rainfed
        # Criteria of irrigated and rainfed cropland classification
        # More than 10% (fraction 0.1) rainfed 30m pixels in a 2km pixel will be classified
        # as "Rainfed cropland". Also, it should have <6% tree cover.
        rainfed_frac_threshold = 0.10
        tree_threshold = 6  # unit in %

        # # Irrigated
        # A 2km pixel with >2% irr fraction was used to classify as irrigated
        irrigated_frac_threshold_for_irrigated_class = 0.02

        # list of years_list when there are both irrigated and rainfed fraction datasets derived from
        # IrrMapper/LANID and USDA CDL. Classifying those data with defined threshold
        years_with_both_irrigated_rainfed_frac_data = [2008, 2009, 2010, 2011, 2012, 2013, 2014,
                                                       2015, 2016, 2017, 2018, 2019, 2020]

        tree_cover_dataset = glob(os.path.join(tree_cover_dir, '*.tif'))[0]
        tree_arr = read_raster_arr_object(tree_cover_dataset, get_file=False)

        for year in years_with_both_irrigated_rainfed_frac_data:
            print(f'Classifying rainfed and irrigated cropland data for year {year}')

            rainfed_frac_data = os.path.join(rainfed_fraction_dir, f'Rainfed_Frac_{year}.tif')
            irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')

            rain_arr, rain_file = read_raster_arr_object(rainfed_frac_data)
            irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

            # classification using defined rainfed, irrigated fraction, and tree fraction threshold. -9999 is no data
            rainfed_cropland = np.where((rain_arr >= rainfed_frac_threshold) &
                                        (tree_arr <= tree_threshold), 1, -9999)

            irrigated_cropland = np.where((irrig_arr > irrigated_frac_threshold_for_irrigated_class), 1, -9999)

            # saving classified data
            output_rainfed_cropland_raster = os.path.join(rainfed_cropland_output_dir, f'Rainfed_cropland_{year}.tif')
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir,
                                                            f'Irrigated_cropland_{year}.tif')

            write_array_to_raster(raster_arr=rainfed_cropland, raster_file=rain_file, transform=rain_file.transform,
                                  output_path=output_rainfed_cropland_raster,
                                  dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster,
                                  dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case

        ############################
        # irrigated fraction data is also available for 2000-2007. Classifying those data to
        # irrigated cropland with defined threshold
        years_rest_irrigated_frac_data = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]

        for year in years_rest_irrigated_frac_data:
            print(f'Classifying rainfed and irrigated cropland data for year {year}')

            irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')

            irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

            # classification using defined rainfed and irrigated fraction threshold. -9999 is no data
            irrigated_cropland = np.where(irrig_arr > irrigated_frac_threshold_for_irrigated_class, 1, -9999)

            # saving classified data
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir,
                                                            f'Irrigated_cropland_{year}.tif')
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster,
                                  dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
    else:
        pass


def filter_rainfed_irrigated_cropET_with_rainfed_irrigated_cropland(rainfed_cropland_dir, irrigated_cropland_dir,
                                                                    rainfed_cropET_input_dir,
                                                                    irrigated_cropET_input_dir,
                                                                    rainfed_cropET_output_dir,
                                                                    irrigated_cropET_output_dir,
                                                                    skip_processing=False):
    """
    Filter Irrigated and Rainfed cropET data by rainfed and irrigated cropland, respectively.

    ** The downloaded Irrigated and Rainfed cropET data from GEE is not fully filtered for rainfed and irrigated
    cropland because in some pixels there are some rainfed and some irrigated fields. So, we first classify rainfed and
    irrigated cropland by rainfed and irrigation fraction threshold (detail in classify_irrigated_rainfed_cropland()
    func), then apply the cropland filters to extract cropET on the purely rainfed and irrigated cropland pixels.

    :param rainfed_cropland_dir: Input directory filepath of rainfed cropland data.
    :param irrigated_cropland_dir: Input directory filepath of irrigated cropland data.
    :param rainfed_cropET_input_dir: Input directory filepath of raw rainfed cropET data.
    :param irrigated_cropET_input_dir: Input directory filepath of raw irrigated cropET data.
    :param rainfed_cropET_output_dir: Output directory filepath of filtered rainfed cropET data.
    :param irrigated_cropET_output_dir: Output directory filepath of filtered irrigated cropET data.
    :param skip_processing: Set to True if want to skip filtering irrigated and rainfed cropET data.

    :return: None.
    """
    if not skip_processing:
        makedirs([rainfed_cropET_output_dir, irrigated_cropET_output_dir])

        # cropET datasets have been extracted from openET for the following years_list and months only
        years_to_filter_irrig_cropET = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                        2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                        2016, 2017, 2018, 2019, 2020]
        years_to_filter_rainfed_cropET = [2008, 2009, 2010, 2011, 2012, 2013, 2014,
                                          2015, 2016, 2017, 2018, 2019, 2020]
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

        for year in years_to_filter_rainfed_cropET:
            print(f'Filtering rainfed cropET data for year {year}...')

            # pure rainfed cropland filtered using rainfed fraction threshold
            # (rainfed frac > 0.10). Tree cover is less than 6%
            rainfed_cropland_data = glob(os.path.join(rainfed_cropland_dir, f'*{year}*.tif'))[0]
            rainfed_cropland_arr = read_raster_arr_object(rainfed_cropland_data, get_file=False)

            for month in months_to_filter_cropET:
                # # applying rainfed cropland filter to get cropET at purely rainfed pixels
                rainfed_cropET_data = glob(os.path.join(rainfed_cropET_input_dir, f'*{year}_{month}*.tif'))[0]
                rainfed_cropET_arr, rainfed_cropET_file = read_raster_arr_object(rainfed_cropET_data)

                # applying the filter
                rainfed_cropET_arr[np.isnan(rainfed_cropland_arr)] = -9999

                filtered_output_raster = os.path.join(rainfed_cropET_output_dir, f'Rainfed_cropET_{year}_{month}.tif')
                write_array_to_raster(raster_arr=rainfed_cropET_arr, raster_file=rainfed_cropET_file,
                                      transform=rainfed_cropET_file.transform, output_path=filtered_output_raster)
    else:
        pass


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


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, output_dir_prism_yearly=None,
                       year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       keyword='prism_precip',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
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
                    Default set to 'prism_precip'.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: None.
    """

    if not skip_processing:
        interim_dir_for_monthly_data = os.path.join(output_dir_prism_monthly, 'interim_dir_for_monthly_data')

        if output_dir_prism_yearly is not None:
            makedirs([output_dir_prism_monthly, output_dir_prism_yearly, interim_dir_for_monthly_data])
        else:
            makedirs([output_dir_prism_monthly, interim_dir_for_monthly_data])

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
            # scale original datasets to 2km resolutions and then clipping them at reference raster (Western US) extent
            interim_monthly_raster = clip_resample_reproject_raster(input_raster=data,
                                                                    input_shape=west_US_shape,
                                                                    raster_name=monthly_raster_name, keyword=' ',
                                                                    output_raster_dir=interim_dir_for_monthly_data,
                                                                    clip=False, resample=True, clip_and_resample=False,
                                                                    targetaligned=True, resample_algorithm='near',
                                                                    use_ref_width_height=False, ref_raster=None,
                                                                    resolution=resolution)

            clip_resample_reproject_raster(input_raster=interim_monthly_raster,
                                           input_shape=west_US_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_prism_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=False, ref_raster=ref_raster,
                                           resolution=resolution)
        #########
        # # Code-block for summing monthly precipitation data for years_list
        #########
        for year in year_list:  # first loop for years_list
            print(f'Processing {keyword} data for {year}...')

            if 'precip' in keyword:
                prism_datasets = glob(
                    os.path.join(output_dir_prism_monthly, f'*{year}*.tif'))  # monthly prism datasets for each year

                # Summing raster for each year
                summed_output_for_year = os.path.join(output_dir_prism_yearly, f'prism_precip_{year}.tif')
                sum_rasters(raster_list=prism_datasets, raster_dir=None, output_raster=summed_output_for_year,
                            ref_raster=prism_datasets[0])

            elif any(i in keyword for i in ['tmax', 'tmin']):

                print('This code does not aggregate prism tmax and tmin data for entire year/growing season')

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


def sum_OpenET_yearly_data(year_list, input_OpenET_monthly_dir, output_dir_OpenET_yearly,
                           output_dir_OpenET_growing_season,
                           skip_processing=False):
    """
    Process (sum for Western US extent) OpenET ET datasets for a year and the year's growing season (April to october).

    :param year_list: Tuple/list of years_list for which to process data.
    :param input_OpenET_monthly_dir: Directory file path of downloaded OpenET ensemble monthly datasets.
    :param output_dir_OpenET_yearly: File path of directory to save summed openET data for each year.
    :param output_dir_OpenET_growing_season: File path of directory to save summed openET data for each year's
                                     growing season at Western US extent.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_OpenET_yearly, output_dir_OpenET_growing_season])

        for year in year_list:  # first loop for years_list
            # # for total years_list
            print(f'summing OpenET data for year {year}...')
            openet_datasets = glob(os.path.join(input_OpenET_monthly_dir, f'*{year}*.tif'))

            # Summing raster for each growing season
            summed_output_for_year = os.path.join(output_dir_OpenET_yearly, f'OpenET_ensemble_{year}.tif')
            sum_rasters(raster_list=openet_datasets, raster_dir=None, output_raster=summed_output_for_year,
                        ref_raster=openet_datasets[0])
    else:
        pass


def sum_GridMET_RET_yearly_data(input_RET_monthly_dir, output_dir_RET_yearly, output_dir_RET_growing_season,
                                year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                           2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                skip_processing=False):
    """
    Process (sum for Western US extent) GridMET RET datasets for for a year and the year's growing season
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


def sum_cropET_water_yr(years_list, input_cropET_monthly_dir, output_dir_water_yr,
                        save_keyword, skip_processing=False):
    """
    Process (sum for Western US extent) irrigated/rainfed cropET for water year.


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
            print(et_water_yr_list)
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


def process_AWC_data(input_dir, westUS_shape, output_dir, ref_raster=WestUS_raster, resolution=model_res,
                     skip_processing=False):
    """
    Process available water capacity (AWC) data for Western US

    :param input_dir: Filepath of input directory.
    :param westUS_shape: Filepath of Western US shapefile.
    :param output_dir: Filepath of output directory.
    :param ref_raster: Fileapth of Western US reference raster.
    :param resolution: Model resolution.
    :param skip_processing: Set to True to skip the process.

    :return: None
    """
    if not skip_processing:
        makedirs([output_dir])

        AWC_raster = glob(os.path.join(input_dir, '*.tif'))[0]

        clip_resample_reproject_raster(input_raster=AWC_raster, input_shape=westUS_shape, output_raster_dir=output_dir,
                                       raster_name='AWC.tif', clip_and_resample=True, resolution=resolution,
                                       ref_raster=ref_raster)


def develop_excess_ET_filter(years_list, water_yr_precip_dir, water_yr_rainfed_ET_dir, output_dir,
                             skip_processing=False):
    """
    Developing a yearly filter for rainfed cropET (effective precip) training data. Using this filter, we will exclude
    pixels where total rainfed crop ET in a water year is higher than precipitation of that water year
    (precipitation from from last year. These filtered out pixels that are using more water from storage than precipitation.
    But we only want to consider pixels where ET mostly comes from precipitation, with some supplement from storage that
    has been built from precipitation over the growing season of that particular year. In addition, this will help keep
    water year Peff / water year precip ratio < 1.

    :param years_list: List of years_list to process the data for.
    :param water_yr_precip_dir: Input directory of water year precip data
    :param water_yr_rainfed_ET_dir: Input directory of water year rainfed crop ET data.
    :param output_dir: Filepath of output directory to save processed data.
    :param skip_processing: Set to True if want to skip processing this dataset.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # we will compare growing season ET for a year with previous year's total precip.
        # so, there is a year lag in precip_years list
        for yr in years_list:
            print(f'processing Excess_ET_filter data for year {yr}')

            # getting water year precip data
            precip_data = glob(os.path.join(water_yr_precip_dir, f'*{yr}*.tif'))[0]
            precip_arr = read_raster_arr_object(precip_data, get_file=False)

            # getting growing season et data
            et_data = glob(os.path.join(water_yr_rainfed_ET_dir, f'*{yr}*.tif'))[0]
            et_arr, file = read_raster_arr_object(et_data)

            # setting value 1 to pixels where water year's total precip is greater than this year's
            # total growing season ET
            # we will only take training rainfed cropET (effective precip) data where the values are 1
            new_arr = np.where((precip_arr < et_arr) | np.isnan(et_arr), -9999, 1)

            output_raster = os.path.join(output_dir, f'Excess_ET_filter_{yr}.tif')
            write_array_to_raster(raster_arr=new_arr, raster_file=file, transform=file.transform,
                                  output_path=output_raster, dtype=np.float32)
    else:
        pass


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

            # clipping and resampling the growing season data with the western US reference raster
            interim_raster = clip_resample_reproject_raster(input_raster=data,
                                                            input_shape=WestUS_shape,
                                                            raster_name=raster_name,
                                                            output_raster_dir=interim_dir,
                                                            clip=False, resample=False, clip_and_resample=True,
                                                            targetaligned=True, resample_algorithm='near',
                                                            use_ref_width_height=False, ref_raster=None,
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


def dynamic_gs_sum_ET(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                      sum_keyword, skip_processing=False):
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


def dynamic_gs_sum_peff(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                        sum_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) sums effective precipitation (peff) monthly rasters for
    the growing seasons (with 3 month's lag peff included before the gorwing season starts).

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
        makedirs([gs_output_dir])

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern_current_yr = re.compile(r'_([0-9]{1,2})\.tif')
        month_pattern_prev_yr = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            print(f'Dynamically summing {sum_keyword} monthly datasets for growing season {year}...')

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
            output_name = f'{sum_keyword}_{year}.tif'
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


def filter_effective_precip_training_data(training_zone_shp, general_output_dir, refraster=WestUS_raster,
                                          resolution=model_res, skip_processing=False):
    """
    Filter the rainfed cropET training data by defined filters and bounding boxes.

    :param training_zone_shp: Input shapefile path of the training bounding box zones.
    :param general_output_dir: Filepath of general output directory. The final output dir will be configured inside
                                the function.
    :param refraster: Default set to Western US reference raster.
    :param resolution: Default set to model resolution.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        print('Filtering effective precip training data...')

        # years_list and months to process over
        # starting from 2008 as rainfed cropET dataset starts from 2008
        years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

        # training zone bounding box gdf
        training_zone_gdf = gpd.read_file(training_zone_shp)

        # reference raster
        ref_arr, ref_file = read_raster_arr_object(refraster)
        total_bounds = ref_file.bounds

        # primary and secondary output directory creation
        shape_temp_output_dir = os.path.join(general_output_dir, 'shapes')
        cropET_interim_shape_dir = os.path.join(general_output_dir, 'cropET_interim_for_shapes')
        final_filtered_cropET_dir = os.path.join(general_output_dir, 'final_filtered_cropET_for_training')

        # removing old directory and making a new one
        if os.path.exists(general_output_dir):
            shutil.rmtree(general_output_dir)

        makedirs([general_output_dir, shape_temp_output_dir, cropET_interim_shape_dir, final_filtered_cropET_dir])

        for id, geom in zip(training_zone_gdf['id'], training_zone_gdf['geometry']):
            # creating shapefile for each of the training boxes and rasterizing the shapefile such that values inside
            # the box/shapefile is 1 and outside the box/shapefile is no data for that respective box

            # creating polygon shapefile for each training zone
            zone_selected = training_zone_gdf[training_zone_gdf['id'] == id]

            zone_shapefile = os.path.join(shape_temp_output_dir, f'shape_{id}.shp')
            zone_selected.to_file(zone_shapefile)

            # rasterizing each training zone
            zone_raster = os.path.join(shape_temp_output_dir, f'zone_raster_{id}.tif')
            raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                                   outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                                   noData=no_data_value, burnValues=1, attribute=None, allTouched=True)
            gdal.Rasterize(destNameOrDestDS=zone_raster, srcDS=zone_shapefile, options=raster_options,
                           resolution=resolution)

        rainfed_cropET_dir = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly'
        rainfed_cropland_dir = '../../Data_main/Raster_data/Rainfed_cropland'
        irrigated_cropland_dir = '../../Data_main/Raster_data/Irrigated_cropland'
        usda_cdl_dir = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'
        excess_ET_filter_dir = '../../Data_main/Raster_data/Excess_ET_filter'
        slope_dir = '../../Data_main/Raster_data/Slope/WestUS'

        # collecting all rainfed cropET dataset
        rainfed_cropET_datasets = []
        for yr in years:
            if yr == 2008:  # for 2008 will only consider October-December months' rainfed cropET as we start considering water year from 2009
                for mn in range(10, 13):
                    rainfed_cropET_datasets.extend(glob(os.path.join(rainfed_cropET_dir, f'*{yr}_{mn}*.tif')))

            elif yr == 2020:  # for 2020 will only consider January-September months' rainfed cropET as we end considering water year at 2020
                for mn in range(1, 10):
                    rainfed_cropET_datasets.extend(glob(os.path.join(rainfed_cropET_dir, f'*{yr}_{mn}*.tif')))

            else:  # for other years_list consider all months
                rainfed_cropET_datasets.extend(glob(os.path.join(rainfed_cropET_dir, f'*{yr}*.tif')))

        # going over each rainfed cropET data and applying the rasterized zone raster on it
        # this will create individual raster for each cropET raster for each bounding box
        for cropET in rainfed_cropET_datasets:
            cropET_arr, file = read_raster_arr_object(cropET)

            year = int(os.path.basename(cropET).split('_')[2])
            month = int(os.path.basename(cropET).split('_')[3].split('.')[0])

            rainfed_cropland_data = glob(os.path.join(rainfed_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            cdl_data = glob(os.path.join(usda_cdl_dir, f'*{year}*.tif'))[0]
            slope_data = glob(os.path.join(slope_dir, '*.tif'))[0]

            # selecting excess ET filter based on water year of the monthly rainfed cropland ET data
            if month in list(range(1, 10)):  # January-September, use excess ET filter of the same water year
                excess_et_filter_data = glob(os.path.join(excess_ET_filter_dir, f'*{year}*.tif'))[0]
            elif month in list(
                    range(10, 13)) and year != 2020:  # October-December, use excess ET filter of the next water year
                excess_et_filter_data = glob(os.path.join(excess_ET_filter_dir, f'*{year + 1}*.tif'))[0]

            rainfed_cropland_arr = read_raster_arr_object(rainfed_cropland_data, get_file=False)
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)
            cdl_arr = read_raster_arr_object(cdl_data, get_file=False)
            excess_et_arr = read_raster_arr_object(excess_et_filter_data, get_file=False)
            slope_arr = read_raster_arr_object(slope_data, get_file=False)

            # grass/pasture lands with rainfed croplands or any rainfed cropland with no overlapping with irrigated croplands,
            # excess_et_filter = 1 in both cases

            # collecting the train zone rasters and looping over them to apply each train zone raster on a cropET raster
            zone_rasters = glob(os.path.join(shape_temp_output_dir, '*.tif'))

            for zone_ras in zone_rasters:
                zone_id = os.path.basename(zone_ras).split('_')[2].split('.')[0]
                zone_arr = read_raster_arr_object(zone_ras, get_file=False)

                filtered_cropET_arr = np.where(((cdl_arr == 176) & (rainfed_cropland_arr == 1) & (excess_et_arr == 1)) |
                                               ((rainfed_cropland_arr == 1) &
                                                # (irrigated_cropland_arr == -9999) &
                                                (excess_et_arr == 1)), cropET_arr, -9999)
                filtered_cropET_arr = np.where(slope_arr <= 1, filtered_cropET_arr, -9999)

                # applying the zone raster on the filtered cropET so that everything outside the box becomes nodata
                final_cropET_arr = filtered_cropET_arr * zone_arr

                # saving the cropET raster for each year-month for each train zone
                output_cropET = os.path.join(cropET_interim_shape_dir, f'cropET_zone_{zone_id}_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_cropET_arr, raster_file=file, transform=file.transform,
                                      output_path=output_cropET)

        # compiling final filtered training data
        for year in years:
            if year == 2008:
                months = range(10, 13)
            elif year == 2020:
                months = range(1, 10)
            else:
                months = range(1, 13)

            for month in months:
                print(f'compiling filtered effective precip training data for year {year}, month {month}...')

                # collecting data for all boxes for a single year-month
                # the  dot (.) is important to collect data for the same months. Not providing 'tif' here as glob will find
                # the rasters without it and we can't use both 'dot' and 'tif' for our purpose at the same time.
                filtered_training_rasters = glob(os.path.join(cropET_interim_shape_dir, f'*{year}_{month}.*'))
                final_train_arr = None  # we will iterate and paste values of each training box here

                # compiling the individual zone cropET raster in a single raster for each year-month
                for each in filtered_training_rasters:
                    if each == filtered_training_rasters[0]:
                        arr = read_raster_arr_object(each, get_file=False)
                        final_train_arr = arr

                    else:
                        arr = read_raster_arr_object(each, get_file=False)
                        final_train_arr = np.where(~np.isnan(final_train_arr), final_train_arr, arr)

                output_raster_name = os.path.join(final_filtered_cropET_dir, f'CropET_for_training_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_train_arr, raster_file=ref_file, transform=ref_file.transform,
                                      output_path=output_raster_name)
    else:
        pass


def accumulate_monthly_datasets_to_water_year(skip_processing=False):
    """
    accumulates monthly datasets to water year by sum or mean.

    :param skip_processing: Set to true to skip this processing step.

    :return: False.
    """
    monthly_data_path_dict = {
        'Effective_precip_train': '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/final_filtered_cropET_for_training',
        'PRISM_Tmax': '../../Data_main/Raster_data/PRISM_Tmax/WestUS_monthly',
        'PRISM_Tmin': '../../Data_main/Raster_data/PRISM_Tmin/WestUS_monthly',
        'GRIDMET_Precip': '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_monthly',
        'GRIDMET_RET': '../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly',
        'GRIDMET_vap_pres_def': '../../Data_main/Raster_data/GRIDMET_vap_pres_def/WestUS_monthly',
        'GRIDMET_max_RH': '../../Data_main/Raster_data/GRIDMET_max_RH/WestUS_monthly',
        'GRIDMET_min_RH': '../../Data_main/Raster_data/GRIDMET_min_RH/WestUS_monthly',
        'GRIDMET_wind_vel': '../../Data_main/Raster_data/GRIDMET_wind_vel/WestUS_monthly',
        'GRIDMET_short_rad': '../../Data_main/Raster_data/GRIDMET_short_rad/WestUS_monthly',
        'DAYMET_sun_hr': '../../Data_main/Raster_data/DAYMET_sun_hr/WestUS_monthly',
        'TERRACLIMATE_SR': '../../Data_main/Raster_data/TERRACLIMATE_SR/WestUS_monthly',
        'Rainy_days': '../../Data_main/Raster_data/Rainy_days/WestUS_monthly'}

    water_yr_accum_dict = {
        'Effective_precip_train': 'sum',
        'Irrigated_cropET': 'sum',
        'PRISM_Tmax': 'mean',
        'PRISM_Tmin': 'mean',
        'GRIDMET_Precip': ['sum', 'mean'],
        'GRIDMET_RET': 'sum',
        'GRIDMET_vap_pres_def': 'mean',
        'GRIDMET_max_RH': 'mean',
        'GRIDMET_min_RH': 'mean',
        'GRIDMET_wind_vel': 'mean',
        'GRIDMET_short_rad': 'mean',
        'DAYMET_sun_hr': 'mean',
        'TERRACLIMATE_SR': ['sum', 'mean'],
        'Rainy_days': 'sum'}

    if not skip_processing:
        # # processing the variables/predictors
        for var, path in monthly_data_path_dict.items():
            print(f'accumulating monthly datasets to water year for {var}...')

            # creating output_dir
            if var == 'Effective_precip_train':
                output_dir = os.path.join(os.path.dirname(path), 'final_filtered_cropET_for_training_water_year')
                years_to_run = range(2009, 2020 + 1)

            elif var == 'Irrigated_cropET':
                output_dir = os.path.join(os.path.dirname(path), 'WestUS_water_year')
                years_to_run = range(2000, 2020 + 1)

            else:
                output_dir = os.path.join(os.path.dirname(path), 'WestUS_water_year')
                years_to_run = range(2000, 2020 + 1)

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


def fraction_SR_precip_water_yr(years_list, input_dir_runoff, input_dir_precip, output_dir, nodata=no_data_value,
                                skip_processing=False):
    """
    Estimate water year fraction of surface runoff to precipitation.

    :param years_list: List of years to process data for.
    :param input_dir_runoff: Filepath of water year summed surface runoff data directory.
    :param input_dir_precip: Filepath of water year summed precipitation data directory.
    :param output_dir: Filepath of output directory.
    :param nodata: No data value. Default set to -9999.
    :param skip_processing: Set to true if want to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        for year in years_list:
            print(f'estimating water year runoff/precipitation fraction for year {year}...')

            # loading and reading datasets
            sr_data = glob(os.path.join(input_dir_runoff, f'*{year}*.tif'))[0]
            precip_data = glob(os.path.join(input_dir_precip, f'*{year}*.tif'))[0]

            sr_arr, raster_file = read_raster_arr_object(sr_data)
            precip_arr = read_raster_arr_object(precip_data, get_file=False)

            # calculating runoff/precip fraction for the water year
            frac_arr = np.where((sr_arr != -9999) & (precip_arr != -9999), sr_arr / precip_arr, nodata)

            # saving estimated raster
            output_raster = os.path.join(output_dir, f'Runoff_precip_fraction_{year}.tif')
            write_array_to_raster(frac_arr, raster_file, raster_file.transform, output_raster)

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


def estimate_PET_by_P_water_yr(years_list, input_dir_PET, input_dir_precip, output_dir, nodata=no_data_value,
                               skip_processing=False):
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


def process_Ksat_data_for_WestUS(ksat_data, output_dir, skip_processing=False):
    """
    Process saturated hydraulic conductivity data for the Western US. Source data comes from
    Gupta et al. 2021 (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002242).
    Output dataset's unit is in cm/day (the same as the input dataset).

    :param ksat_data: Filepath of Ksat data.
    :param westUS_shapefile: Filepath of Western US shapefile
    :param output_dir: Filepath of output dir.
    :param skip_processing: Set to True if want to skip this process.

    :return: None.
    """
    if not skip_processing:
        print('clipping and resampling saturated hydraulic conductivity data for the Western US')

        clip_resample_reproject_raster(input_raster=ksat_data, input_shape=WestUS_shape,
                                       raster_name='Ksat_0cm.tif', keyword=' ',
                                       output_raster_dir=output_dir,
                                       clip=False, resample=False, clip_and_resample=True,
                                       targetaligned=True, resample_algorithm='near',
                                       use_ref_width_height=False,
                                       ref_raster=WestUS_raster,
                                       resolution=model_res)
    else:
        pass


def create_rel_infiltration_capacity_dataset(years_list, ksat_data, precip_intensity_dir, output_dir, skip_processing):
    """
    Create a water year relative infiltration capacity (saturated hydraulic conductivity/precipitation intensity)
    dataset for the western US.

    *** Higher values would indicate that the soil is capable of infiltrating water more effectively than
    the rate of precipitation.

    :param years_list: List of years to process data for.
    :param ksat_data: Filepath of Ksat data.
    :param precip_intensity_dir: Filepath of precipitation intensity dataset directory.
    :param output_dir: Filepath of output dir.
    :param skip_processing: Set to True if want to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        for year in years_list:
            print(f'creating relative infiltration capacity dataset for year {year}...')

            # loading and reading datasets
            precip_intensity_data = glob(os.path.join(precip_intensity_dir, f'*{year}*.tif'))[0]
            precip_intensity_arr = read_raster_arr_object(precip_intensity_data, get_file=False)

            ksat_arr, raster_file = read_raster_arr_object(ksat_data)

            # estimating relative infiltration capacity
            rel_infil_arr = np.where(ksat_arr != -9999, ksat_arr / precip_intensity_arr, -9999)

            # saving output raster
            output_raster = os.path.join(output_dir, f'rel_infiltration_capacity_{year}.tif')
            write_array_to_raster(rel_infil_arr, raster_file, raster_file.transform, output_raster)


def develop_P_PET_correlation_dataset(monthly_precip_dir, monthly_pet_dir, output_dir,skip_processing=False):
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


def run_all_preprocessing(skip_process_GrowSeason_data=False,
                          skip_prism_processing=False,
                          skip_gridmet_precip_processing=False,
                          skip_gridmet_RET_precessing=False,
                          skip_merging_rainfed_frac=False,
                          skip_merging_rainfed_cropET=False,
                          skip_merging_irrigated_frac=False,
                          skip_merging_irrigated_cropET=False,
                          skip_classifying_irrigated_rainfed_cropland=False,
                          skip_filtering_irrigated_rainfed_cropET=False,
                          skip_summing_irrigated_cropET_gs=False,
                          skip_summing_rainfed_cropET_gs=False,
                          skip_summing_rainfed_cropET_water_yr=False,
                          skip_sum_openET=False,
                          skip_excess_ET_filter_processing=False,
                          skip_processing_slope_data=False,
                          skip_process_AWC_data=False,
                          skip_effective_precip_training_data_filtering=False,
                          skip_accum_to_water_year_datasets=False,
                          skip_summing_irrigated_cropET_water_yr=False,
                          skip_estimate_runoff_precip_frac=False,
                          skip_estimate_precip_intensity=False,
                          skip_estimate_dryness_index=False,
                          skip_estimate_peff_water_yr_frac=False,
                          skip_process_ksat_data=False,
                          skip_process_rel_infiltration_capacity_data=False,
                          skip_create_P_PET_corr_dataset=False,
                          ref_raster=WestUS_raster):
    """
    Run all preprocessing steps.

    :param skip_process_GrowSeason_data: Set to True to skip processing growing season data.
    :param skip_prism_processing: Set True if want to skip prism (precipitation and temperature) data preprocessing.
    :param skip_gridmet_precip_processing: Set True to skip gridmet precip yearly data processing.
    :param skip_gridmet_RET_precessing: Set to True to skip GridMET RET data processing.
    :param skip_merging_rainfed_frac: Set to True to skip merging rainfed fraction data.
    :param skip_merging_rainfed_cropET: Set to True to skip merging rainfed cropET data.
    :param skip_merging_irrigated_frac: Set to True to skip merging irrigated fraction data.
    :param skip_merging_irrigated_cropET: Set to True to skip merging irrigated fraction data.
    :param skip_classifying_irrigated_rainfed_cropland: Set to True if want to skip classifying irrigated and
                                                        rainfed cropland data.
    :param skip_filtering_irrigated_rainfed_cropET: Set to True if want to skip filtering irrigated and rainfed cropET
                                                    data.
    :param skip_summing_irrigated_cropET_gs: Set to True if want to skip summing irrigated cropET data summing for year/grow season.
    :param skip_summing_rainfed_cropET_gs: Set to True if want to skip summing rainfed cropET for growing season.
    :param skip_summing_rainfed_cropET_water_yr: Set to True if want to skip summing rainfed cropET for water year.
    :param skip_sum_openET: Set True to skip openET ensemble yearly data processing.
    :param skip_excess_ET_filter_processing: Set to True if want to skip excess ET filter dataset processing.
    :param skip_processing_slope_data: Set to True if want to skip DEM to slope conversion.
    :param skip_process_AWC_data: Set to True ti skip processing AWC data.
    :param skip_effective_precip_training_data_filtering: Set to True if want to skip filtering.
    :param skip_accum_to_water_year_datasets: Set to True to skip accumulating monthly dataset to water year.
    :param skip_summing_irrigated_cropET_water_yr: Set to True if want to skip summing irrigated cropET for water year.
    :param skip_estimate_runoff_precip_frac: Set to True to skip processing water year runoff/precipitation fraction data.
    :param skip_estimate_precip_intensity: Set to True to skip processing water year precipitation intensity data.
    :param skip_estimate_dryness_index: Set to True to skip processing water year PET/P (dryness Index) data.
    :param skip_estimate_peff_water_yr_frac: Set to True if want to skip water year Peff/water year precip fraction
                                             estimation for annual scale model.
    :param skip_process_ksat_data: Set to True to skip processing saturated hydraulic conductivity data.
    :param skip_process_rel_infiltration_capacity_data: Set to True to skip processing relative infiltration capacity data.
    :param skip_create_P_PET_corr_dataset: Set to True to skip create P-PET correlation dataset.
    :param ref_raster: Filepath of Western US reference raster to use in 2km pixel lat-lon raster creation and to use
                        as reference raster in other processing operations.

    :return: None.
    """
    # process growing season data
    extract_month_from_GrowSeason_data(GS_data_dir='../../Data_main/Raster_data/Growing_season',
                                       skip_processing=skip_process_GrowSeason_data)

    # merge rainfed fraction dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(year_list=(2008, 2009, 2010, 2011, 2012,
                                                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                                   input_dir_irrmapper='../../Data_main/Raster_data/Rainfed_Frac_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Rainfed_Frac_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac',
                                                   merge_keyword='Rainfed_Frac', monthly_data=False,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_rainfed_frac)

    # merge rainfed cropET dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(year_list=(2008, 2009, 2010, 2011, 2012,
                                                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                                   input_dir_irrmapper='../../Data_main/Raster_data/Rainfed_crop_OpenET_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Rainfed_crop_OpenET_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly_raw',
                                                   merge_keyword='Rainfed_cropET', monthly_data=True,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_rainfed_cropET)

    # merge irrigated fraction dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                                              2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                                              2016, 2017, 2018, 2019, 2020),
                                                   input_dir_irrmapper='../../Data_main/Raster_data/Irrigation_Frac_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Irrigation_Frac_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
                                                   merge_keyword='Irrigated_Frac', monthly_data=False,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_irrigated_frac)

    # merge irrigated cropET dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                                              2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                                              2016, 2017, 2018, 2019, 2020),
                                                   input_dir_irrmapper='../../Data_main/Raster_data/Irrig_crop_OpenET_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Irrig_crop_OpenET_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly_raw',
                                                   merge_keyword='Irrigated_cropET', monthly_data=True,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_irrigated_cropET)

    # classify rainfed and irrigated cropland data
    classify_irrigated_rainfed_cropland(
        rainfed_fraction_dir='../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac',
        irrigated_fraction_dir='../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
        tree_cover_dir='../../Data_main/Raster_data/Tree_cover/WestUS',
        rainfed_cropland_output_dir='../../Data_main/Raster_data/Rainfed_cropland',
        irrigated_cropland_output_dir='../../Data_main/Raster_data/Irrigated_cropland',
        skip_processing=skip_classifying_irrigated_rainfed_cropland)

    # filtering rainfed and irrigated cropET with rainfed and irrigated cropland data
    filter_rainfed_irrigated_cropET_with_rainfed_irrigated_cropland(
        rainfed_cropland_dir='../../Data_main/Raster_data/Rainfed_cropland',
        irrigated_cropland_dir='../../Data_main/Raster_data/Irrigated_cropland',
        rainfed_cropET_input_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly_raw',
        irrigated_cropET_input_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly_raw',
        rainfed_cropET_output_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly',
        irrigated_cropET_output_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly',
        skip_processing=skip_filtering_irrigated_rainfed_cropET)

    # sum monthly irrigated cropET for dynamic growing season
    dynamic_gs_sum_ET(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                 2016, 2017, 2018, 2019, 2020),
                      growing_season_dir='../../Data_main/Raster_data/Growing_season',
                      monthly_input_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly',
                      gs_output_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season',
                      sum_keyword='Irrigated_cropET',
                      skip_processing=skip_summing_irrigated_cropET_gs)

    # sum monthly rainfed cropET for dynamic growing season
    dynamic_gs_sum_ET(year_list=(2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                 2016, 2017, 2018, 2019, 2020),
                      growing_season_dir='../../Data_main/Raster_data/Growing_season',
                      monthly_input_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly',
                      gs_output_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_grow_season',
                      sum_keyword='Rainfed_cropET',
                      skip_processing=skip_summing_rainfed_cropET_gs)

    # sum monthly rainfed cropET for water year
    sum_cropET_water_yr(years_list=(2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                    2016, 2017, 2018, 2019, 2020),
                        input_cropET_monthly_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly',
                        output_dir_water_yr='../../Data_main/Raster_data/Rainfed_cropET/WestUS_water_year',
                        save_keyword='Rainfed_cropET',
                        skip_processing=skip_summing_rainfed_cropET_water_yr)

    # prism precipitation data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/Raster_data/PRISM_Precip/bil_format',
                       prism_tif_dir='../../Data_main/Raster_data/PRISM_Precip/tif_format',
                       output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly',
                       output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_Precip/WestUS_yearly',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_precip', skip_processing=skip_prism_processing)

    # prism maximum temperature data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/Raster_data/PRISM_Tmax/bil_format',
                       prism_tif_dir='../../Data_main/Raster_data/PRISM_Tmax/tif_format',
                       output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_Tmax/WestUS_monthly',
                       output_dir_prism_yearly=None,
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_tmax', skip_processing=skip_prism_processing)

    # prism minimum temperature data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/Raster_data/PRISM_Tmin/bil_format',
                       prism_tif_dir='../../Data_main/Raster_data/PRISM_Tmin/tif_format',
                       output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_Tmin/WestUS_monthly',
                       output_dir_prism_yearly=None,
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_tmin', skip_processing=skip_prism_processing)

    # gridmet precip yearly data processing
    sum_GridMET_precip_yearly_data(
        year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                   2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
        input_gridmet_monthly_dir='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_monthly',
        output_dir_yearly='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_yearly',
        skip_processing=skip_gridmet_precip_processing)

    # OpenET ensemble growing season summing
    dynamic_gs_sum_ET(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                 2016, 2017, 2018, 2019, 2020),
                      growing_season_dir='../../Data_main/Raster_data/Growing_season',
                      monthly_input_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_monthly',
                      gs_output_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_grow_season',
                      sum_keyword='OpenET_ensemble',
                      skip_processing=skip_sum_openET)

    # GridMET yearly data processing
    sum_GridMET_RET_yearly_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                           2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                input_RET_monthly_dir='../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly',
                                output_dir_RET_yearly='../../Data_main/Raster_data/GRIDMET_RET/WestUS_yearly',
                                output_dir_RET_growing_season='../../Data_main/Raster_data/GRIDMET_RET/WestUS_grow_season',
                                skip_processing=skip_gridmet_RET_precessing)

    # processing excess ET filter
    develop_excess_ET_filter(years_list=(2009, 2010, 2011, 2012, 2013, 2014,
                                         2015, 2016, 2017, 2018, 2019, 2020),
                             # starting from water year 2009 as 2008 water year couldn't be considered due to missing rainfed cropland data for 2007
                             water_yr_precip_dir='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year/sum',
                             water_yr_rainfed_ET_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_water_year',
                             output_dir='../../Data_main/Raster_data/Excess_ET_filter',
                             skip_processing=skip_excess_ET_filter_processing)

    # converting DEM data to slope
    create_slope_raster(input_raster='../../Data_main/Raster_data/DEM/WestUS/DEM.tif',
                        output_dir='../../Data_main/Raster_data/Slope/WestUS', raster_name='Slope.tif',
                        skip_processing=skip_processing_slope_data)

    # processing available water capacity (AWC) data
    process_AWC_data(input_dir='../../Data_main/Raster_data/Available_water_capacity/awc_gNATSGO',
                     westUS_shape=WestUS_shape,
                     output_dir='../../Data_main/Raster_data/Available_water_capacity/WestUS',
                     ref_raster=ref_raster, resolution=model_res,
                     skip_processing=skip_process_AWC_data)

    # making a latitude longitude raster from reference raster
    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    lon_arr, lat_arr = make_lat_lon_array_from_raster(ref_raster)

    lon_dir = os.path.join('../../Data_main/Raster_data', 'Longitude/WestUS')
    lat_dir = os.path.join('../../Data_main/Raster_data', 'Latitude/WestUS')
    makedirs([lon_dir, lat_dir])

    write_array_to_raster(raster_arr=lon_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=os.path.join(lon_dir, 'Longitude.tif'))
    write_array_to_raster(raster_arr=lat_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=os.path.join(lat_dir, 'Latitude.tif'))

    # filtering effective precipitation training data with excess ET and bounding box filter
    training_zone_shapefile = '../../Data_main/shapefiles/Training_zones/effective_precip_training_zone.shp'
    output_dir = '../../Data_main/Raster_data/Rainfed_cropET_filtered_training'

    filter_effective_precip_training_data(training_zone_shp=training_zone_shapefile,
                                          general_output_dir=output_dir,
                                          refraster=WestUS_raster,
                                          resolution=model_res,
                                          skip_processing=skip_effective_precip_training_data_filtering)

    # # # # # # # # # # # # # # # # # # # # # # for water year model # # # # # # # # # # # # # # # # # # # # # # # # # #

    # accumulating monthly dataset to water year
    accumulate_monthly_datasets_to_water_year(skip_processing=skip_accum_to_water_year_datasets)

    # sum monthly irrigated cropET for water year
    sum_cropET_water_yr(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006,
                                    2007, 2008, 2009, 2010, 2011, 2012, 2013,
                                    2014, 2015, 2016, 2017, 2018, 2019, 2020),
                        input_cropET_monthly_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly',
                        output_dir_water_yr='../../Data_main/Raster_data/Irrigated_cropET/WestUS_water_year',
                        save_keyword='Irrigated_cropET',
                        skip_processing=skip_summing_irrigated_cropET_water_yr)

    # estimate fraction of water year surface runoff to precipitation
    fraction_SR_precip_water_yr(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006,
                                            2007, 2008, 2009, 2010, 2011, 2012, 2013,
                                            2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                input_dir_runoff='../../Data_main/Raster_data/TERRACLIMATE_SR/WestUS_water_year/sum',
                                input_dir_precip='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year/sum',
                                output_dir='../../Data_main/Raster_data/Runoff_precip_fraction',
                                skip_processing=skip_estimate_runoff_precip_frac)

    # estimate water year precipitation intensity (precipitation / rainy days)
    estimate_precip_intensity_water_yr(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006,
                                                   2007, 2008, 2009, 2010, 2011, 2012, 2013,
                                                   2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                       input_dir_precip='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year/mean',
                                       input_dir_rainy_day='../../Data_main/Raster_data/Rainy_days/WestUS_water_year',
                                       output_dir='../../Data_main/Raster_data/Precipitation_intensity',
                                       skip_processing=skip_estimate_precip_intensity)

    # estimate PET/P (dryness index) for water year
    estimate_PET_by_P_water_yr(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006,
                                           2007, 2008, 2009, 2010, 2011, 2012, 2013,
                                           2014, 2015, 2016, 2017, 2018, 2019, 2020),
                               input_dir_PET='../../Data_main/Raster_data/GRIDMET_RET/WestUS_water_year',
                               input_dir_precip='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year/sum',
                               output_dir='../../Data_main/Raster_data/Dryness_index',
                               skip_processing=skip_estimate_dryness_index)

    # process saturated hydraulic conductivity (Ksat) data
    process_Ksat_data_for_WestUS(
        ksat_data='../../Data_main/Raster_data/Saturated_hydraulic_conductivity/raw/Global_Ksat_1Km_s0....0cm_v1.0.tif',
        output_dir='../../Data_main/Raster_data/Saturated_hydraulic_conductivity',
        skip_processing=skip_process_ksat_data)

    # create relative infiltration capacity dataset
    create_rel_infiltration_capacity_dataset(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006,
                                                         2007, 2008, 2009, 2010, 2011, 2012, 2013,
                                                         2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                             ksat_data='../../Data_main/Raster_data/Saturated_hydraulic_conductivity/Ksat_0cm.tif',
                                             precip_intensity_dir='../../Data_main/Raster_data/Precipitation_intensity',
                                             output_dir='../../Data_main/Raster_data/Relative_infiltration_capacity',
                                             skip_processing=skip_process_rel_infiltration_capacity_data)

    # create P-PET correlation dataset
    develop_P_PET_correlation_dataset(monthly_precip_dir='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_monthly',
                                      monthly_pet_dir='../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly',
                                      output_dir='../../Data_main/Raster_data/P_PET_correlation',
                                      skip_processing=skip_create_P_PET_corr_dataset)

    # processing the training data
    # water year rainfed cropET (effective precip) / water year precip fraction estimation
    estimate_peff_precip_water_year_fraction(
        years_list=(2009, 2010, 2011, 2012, 2013, 2014, 2015,
                    2016, 2017, 2018, 2019, 2020),
        peff_dir_water_yr='../../Data_main/Raster_data/Rainfed_cropET_filtered_training/final_filtered_cropET_for_training_water_year',
        precip_dir_water_yr='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year/sum',
        output_dir='../../Data_main/Raster_data/Rainfed_cropET_filtered_training/rainfed_cropET_water_year_fraction',
        skip_processing=skip_estimate_peff_water_yr_frac)
