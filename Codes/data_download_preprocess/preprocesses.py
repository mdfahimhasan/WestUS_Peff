import os
import re
import sys
import datetime
import numpy as np
from glob import glob
from osgeo import gdal
import rasterio as rio

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, mosaic_rasters_list, \
        clip_resample_reproject_raster, sum_rasters, make_lat_lon_array_from_raster


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
    input_raster_names = [f'input_raster_{i}' for i in range(1, len(input_rasters_list)+1)]

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

    # creating an empty array to assign cropland (1) if most year_list between 2008-2011 are cropland
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

    :param year_list: Tuple/list of year_list for which data will be processed.
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

                    if len(total_raster_list) > 0:  # to only merge for year_list and months when data is available
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

                if len(total_raster_list) > 0:  # to only merge for year_list and months when data is available
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
        # More than 30% (fraction 0.3) rainfed and less than 2% irrigated 30m pixels in a 2km pixel will be classified
        # as "Rainfed cropland". Also, it should have <6% tree cover.
        rainfed_frac_threshold = 0.30
        irrigated_frac_threshold_for_rainfed_class = 0.02
        tree_threshold = 6  # unit in %

        # # Irrigated
        # A 2km pixel with any fraction of irrigation will be classified as irrigated.
        # Previously, >2% irr fraction was used to classify as irrigated, which was later removed to cover the
        # boundary pixels in irrigated-agricultural zones. May need post filtering.
        irrigated_frac_threshold_for_irrigated_class = 0

        # list of year_list when there are both irrigated and rainfed fraction datasets derived from
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
                                        ((irrig_arr == irrigated_frac_threshold_for_rainfed_class) | (np.isnan(irrig_arr)))
                                         & (tree_arr <= tree_threshold), 1, -9999)

            irrigated_cropland = np.where((irrig_arr > irrigated_frac_threshold_for_irrigated_class), 1, -9999)

            # saving classified data
            output_rainfed_cropland_raster = os.path.join(rainfed_cropland_output_dir, f'Rainfed_cropland_{year}.tif')
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir, f'Irrigated_cropland_{year}.tif')

            write_array_to_raster(raster_arr=rainfed_cropland, raster_file=rain_file, transform=rain_file.transform,
                                  output_path=output_rainfed_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case

        ############################
        # irrigated fraction data is also available for 2000-2007. Classifying those data to
        # irrigated cropland with defined threshold
        years_rest_irrigated_frac_data = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]

        for year in years_rest_irrigated_frac_data:
            print(f'Classifying rainfed and irrigated cropland data for year {year}')

            irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')

            irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

            # classification using defined rainfed and irrigated fraction threshold. -9999 is no data
            irrigated_cropland = np.where(irrig_arr > irrigated_frac_threshold_for_irrigated_class, 1, -9999)

            # saving classified data
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir, f'Irrigated_cropland_{year}.tif')
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
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
        # cropET datasets have been extracted from openET for the following year_list and months only
        years_to_filter_irrig_cropET = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                        2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                        2016, 2017, 2018, 2019, 2020]
        years_to_filter_rainfed_cropET = [2008, 2009, 2010, 2011, 2012, 2013, 2014,
                                          2015, 2016, 2017, 2018, 2019, 2020]
        months_to_filter_cropET = list(range(1, 13))

        for year in years_to_filter_irrig_cropET:
            print(f'Filtering irrigated cropET data for year {year}...')

            # pure irrigated cropland filtered previously by using irrigated fraction threshold (irrig frac > 0.20)
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)

            for month in months_to_filter_cropET:
                # # applying irrigated cropland filter to get cropET at purely irrigated pixels
                irrigated_cropET_data = glob(os.path.join(irrigated_cropET_input_dir, f'*{year}_{month}*.tif'))[0]
                irrigated_cropET_arr, irrigated_cropET_file = read_raster_arr_object(irrigated_cropET_data)

                # applying the filter
                irrigated_cropET_arr[np.isnan(irrigated_cropland_arr)] = -9999

                filtered_output_raster = os.path.join(irrigated_cropET_output_dir, f'Irrigated_cropET_{year}_{month}.tif')
                write_array_to_raster(raster_arr=irrigated_cropET_arr, raster_file=irrigated_cropET_file,
                                      transform=irrigated_cropET_file.transform, output_path=filtered_output_raster)

        for year in years_to_filter_rainfed_cropET:
            print(f'Filtering rainfed cropET data for year {year}...')

            # pure rainfed cropland filtered previously by using rainfed and irrigated fraction threshold
            # (rainfed frac > 0.20 and irrig frac < 0.02). In both cases, tree cover was less than 6%
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
    :param year_list: Tuple/list of year_list for which prism data was downloaded.
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
        # # Code-block for summing monthly precipitation data for year_list
        #########
        for year in year_list:  # first loop for year_list
            print(f'Processing {keyword} data for {year}...')

            if 'precip' in keyword:
                    prism_datasets = glob(os.path.join(output_dir_prism_monthly, f'*{year}*.tif'))  # monthly prism datasets for each year

                    # Summing raster for each year
                    summed_output_for_year = os.path.join(output_dir_prism_yearly, f'prism_precip_{year}.tif')
                    sum_rasters(raster_list=prism_datasets, raster_dir=None, output_raster=summed_output_for_year,
                                ref_raster=prism_datasets[0])

            elif any(i in keyword for i in ['tmax', 'tmin']):

                    print('This code does not aggregate prism tmax and tmin data for entire year/growing season')

    else:
        pass


def sum_gridmet_precip_yearly_data(year_list, input_gridmet_monthly_dir, output_dir_yearly,
                                   output_dir_grow_season, skip_processing=False):
    """
    Process (sum for Western US extent) GRIDMET Precipitation for a year.

    :param year_list: Tuple/list of year_list for which data will be processed.
    :param input_gridmet_monthly_dir: Directory file path of downloaded gridmet precip monthly datasets.
    :param output_dir_yearly: File path of directory to save summed precip for each year at Western US extent.
    :param output_dir_grow_season: File path of directory to save summed precip for growing season (4-10) of each year
                                   at Western US extent.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_yearly, output_dir_grow_season])

        for year in year_list:  # first loop for year_list
            print(f'summing GRIDMET precip data for year {year}...')
            gridmet_datasets = glob(os.path.join(input_gridmet_monthly_dir, f'*{year}*.tif'))

            # Summing raster for each year
            summed_output_for_year = os.path.join(output_dir_yearly, f'GRIDMET_Precip_{year}.tif')
            sum_rasters(raster_list=gridmet_datasets, raster_dir=None, output_raster=summed_output_for_year,
                        ref_raster=gridmet_datasets[0])

            # Summing raster for each year's growing season
            gridmet_datasets = glob(os.path.join(input_gridmet_monthly_dir, f'*{year}_[4-9]*.tif')) + \
                               glob(os.path.join(input_gridmet_monthly_dir, f'*{year}_10*.tif'))
            summed_output_for_grow_season = os.path.join(output_dir_grow_season, f'GRIDMET_Precip_{year}.tif')
            sum_rasters(raster_list=gridmet_datasets, raster_dir=None, output_raster=summed_output_for_grow_season,
                        ref_raster=gridmet_datasets[0])

    else:
        pass


def sum_OpenET_yearly_data(year_list, input_OpenET_monthly_dir, output_dir_OpenET_yearly,
                           output_dir_OpenET_growing_season,
                           skip_processing=False):
    """
    Process (sum for Western US extent) OpenET ET datasets for a year and the year's growing season (April to october).

    :param year_list: Tuple/list of year_list for which to process data.
    :param input_OpenET_monthly_dir: Directory file path of downloaded OpenET ensemble monthly datasets.
    :param output_dir_OpenET_yearly: File path of directory to save summed openET data for each year.
    :param output_dir_OpenET_growing_season: File path of directory to save summed openET data for each year's
                                     growing season at Western US extent.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_OpenET_yearly, output_dir_OpenET_growing_season])

        for year in year_list:  # first loop for year_list
            # # for total year_list
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
    :param year_list: Tuple/list of year_list for which to process data.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_RET_yearly, output_dir_RET_growing_season])

        for year in year_list:  # first loop for year_list
            # # for total year_list
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


def develop_excess_ET_filter(monthly_precip_dir, output_dir_precip_water_yr, grow_season_ET_dir, output_dir,
                             skip_processing=False):
    """
    Developing a yearly filter for rainfed cropET (effective precip) training data. Using this filter, we will exclude
    pixels where total ET (openET ensemble) in a growing season is higher than precipitation of that water year
    (precipitation from from last year. These
    filtered out pixels that are using more water from storage than precipitation. But we only want to consider pixels
    where ET mostly comes from precipitation, with some supplement from storage that has been built from precipitation
    over the growing season of that particular year.

    :param monthly_precip_dir: Input directory of monthly total precipitation data (the code is written for prism monthly data).
    :param output_dir_precip_water_yr: Output directory to save summed water year precipitation data.
    :param grow_season_ET_dir: Input directory of growing season ET data.
    :param output_dir: Filepath of output directory to save processed data.
    :param skip_processing: Set to True if want to skip processing this dataset.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir_precip_water_yr, output_dir])
        # we will compare growing season ET for a year with previous year's total precip.
        # so, there is a year lag in precip_years list
        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

        for yr in years:
            print(f'procesing Excess_ET_filter data for year {yr}')

            # # summing precipitation for water year (previous year's October to current year's september)
            # the searching operation is based on prism monthly data format
            precip_data_prev_years = glob(os.path.join(monthly_precip_dir, f'*{yr - 1}_1[0-2].*tif'))
            precip_data_current_years = glob(os.path.join(monthly_precip_dir, f'*{yr}_[1-9].*tif'))
            precip_water_yr_list = precip_data_prev_years + precip_data_current_years

            summed_precip_arr, _ = \
                sum_rasters(raster_list=precip_water_yr_list,
                            raster_dir=None,
                            output_raster=os.path.join(output_dir_precip_water_yr, f'prism_precip_water_yr_{yr}.tif'),
                            ref_raster=precip_water_yr_list[0])

            # getting growing season et data for current year
            et_data = glob(os.path.join(grow_season_ET_dir, f'*{yr}*.tif'))[0]
            et_arr, file = read_raster_arr_object(et_data)

            # setting value 1 to pixels where water year's total precip is greater than this year's
            # total growing season ET
            # we will only take training rainfed cropET (effective precip) data where the values are 1
            new_arr = np.where((summed_precip_arr < et_arr) | np.isnan(summed_precip_arr), -9999, 1)

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

        :param year (int): The year in which the DOY is being calculated.
        :param doy (int): The day of the year (1 through 365 or 366 for leap years).

        :return: Month of the corresponding date.
        """
        if np.isnan(doy):  # Check if the DOY is NaN
            return np.nan

        # January 1st of the given year
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


def dynamic_gs_sum_peff_cropET(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                               sum_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) sums effective precipitation and irrigated crop ET monthly rasters for
    the growing seasons.

    :param year_list: List of years to process the data for.
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
            end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)            # band 2

            # We create a 1 pixel "kernel", representing months 1 to 12 (shape - 12, 1, 1).
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
                          skip_summing_irrigated_cropET=False,
                          skip_sum_openET=False,
                          skip_excess_ET_filter_processing=False,
                          skip_processing_slope_data=False,
                          skip_process_AWC_data=False,
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
    :param skip_summing_irrigated_cropET: Set to True if want to skip summing irrigated cropET data summing for year/grow season.
    :param skip_sum_openET: Set True to skip openET ensemble yearly data processing.
    :param skip_excess_ET_filter_processing: Set to True if want to skip excess ET filter dataset processing.
    :param skip_processing_slope_data: Set to True if want to skip DEM to slope conversion.
    :param skip_process_AWC_data: Set to True ti skip processing AWC data.
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
    merge_GEE_data_patches_IrrMapper_LANID_extents(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                                              2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                                              2016, 2017, 2018, 2019, 2020),
                                                   input_dir_irrmapper='../../Data_main/Raster_data/Irrigation_Frac_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Irrigation_Frac_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
                                                   merge_keyword='Irrigated_Frac', monthly_data=False,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_irrigated_frac)

    # merge irrigated cropET dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
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
    dynamic_gs_sum_peff_cropET(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                          2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                          2016, 2017, 2018, 2019, 2020),
                               growing_season_dir='../../Data_main/Raster_data/Growing_season',
                               monthly_input_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly',
                               gs_output_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season',
                               sum_keyword='Irrigated_cropET',
                               skip_processing=skip_summing_irrigated_cropET)

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
    sum_gridmet_precip_yearly_data(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                              2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                   input_gridmet_monthly_dir='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_monthly',
                                   output_dir_yearly='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_yearly',
                                   output_dir_grow_season='../../Data_main/Raster_data/GRIDMET_Precip/WestUS_grow_season',
                                   skip_processing=skip_gridmet_precip_processing)

    # OpenET ensemble growing season summing
    dynamic_gs_sum_peff_cropET(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                          2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                          2016, 2017, 2018, 2019, 2020),
                               growing_season_dir='../../Data_main/Raster_data/Growing_season',
                               monthly_input_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_monthly',
                               gs_output_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_grow_season',
                               sum_keyword='OpenET_ensemble',
                               skip_processing=skip_sum_openET)

    # GridMET yearly data processing
    sum_GridMET_RET_yearly_data(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                           2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                input_RET_monthly_dir='../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly',
                                output_dir_RET_yearly='../../Data_main/Raster_data/GRIDMET_RET/WestUS_yearly',
                                output_dir_RET_growing_season='../../Data_main/Raster_data/GRIDMET_RET/WestUS_grow_season',
                                skip_processing=skip_gridmet_RET_precessing)

    # processing excess ET filter
    develop_excess_ET_filter(monthly_precip_dir='../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly',
                             output_dir_precip_water_yr='../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year',
                             grow_season_ET_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_grow_season',
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