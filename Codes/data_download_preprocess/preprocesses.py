import os
import sys
import numpy as np
from glob import glob
from osgeo import gdal

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, mosaic_rasters_from_directory, \
      mosaic_rasters_list, clip_resample_reproject_raster, sum_rasters, mean_rasters, make_lat_lon_array_from_raster


no_data_value = -9999
model_res = 0.02000000000000000736  # in deg, ~2.22 km
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'


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

    # creating an empty array to assign cropland (1) if most years between 2008-2011 are cropland
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


def merge_GEE_data_patches_IrrMapper_LANID_extents(input_dir_irrmapper, input_dir_lanid, merged_output_dir,
                                                   merge_keyword, monthly_data=True, ref_raster=WestUS_raster,
                                                   skip_processing=False):
    """
    Merge/mosaic downloaded GEE data for IrrMapper and LANID extent.

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

        year_list = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                     2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)

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

                    if len(total_raster_list) > 0:  # to only merge for years and months when data is available
                        merged_raster_name = f'{merge_keyword}_{year}_{month}.tif'
                        mosaic_rasters_list(input_raster_list=total_raster_list, output_dir=merged_output_dir,
                                            raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                            mosaicing_method='first', resolution=model_res, nodata=no_data_value)

                        print(f'{merge_keyword} data merged for year {year}, month {month}')

        else:  # for datasets that are yearly
            for year in year_list:
                search_by = f'*{year}_*.tif'

                # making input raster list by joining rasters of irrmapper extent and rasters of lanid extent
                irrmapper_raster_list = glob(os.path.join(input_dir_irrmapper, search_by))
                lanid_raster_list = glob(os.path.join(input_dir_lanid, search_by))

                irrmapper_raster_list.extend(lanid_raster_list)

                total_raster_list = irrmapper_raster_list

                if len(total_raster_list) > 0:  # to only merge for years and months when data is available
                    merged_raster_name = f'{merge_keyword}_{year}.tif'
                    mosaic_rasters_list(input_raster_list=total_raster_list, output_dir=merged_output_dir,
                                        raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                        mosaicing_method='first', resolution=model_res, nodata=no_data_value)

                    print(f'{merge_keyword} data merged for year {year}')
    else:
        pass


def classify_irrigated_rainfed_cropland(rainfed_fraction_dir, irrigated_fraction_dir,
                                        rainfed_cropland_output_dir, irrigated_cropland_output_dir,
                                        interim_dir_for_maximum_occurrence_rainfed_cropland, skip_processing=False):
    """
    Classifies rainfed and irrigated cropland using rainfed and irrigated fraction data.

    ** The rainfed fraction data is only available for 2008-2020, so rainfed cropland classification can't be done
    directly for 2000-2007. This function classifies rainfed cropland data for 2008-2020 first, then applies
    "maximum occurrence approach" for extending the rainfed cropland dataset for 2000-2007.

    :param rainfed_fraction_dir: Input directory path for rainfed fraction data.
    :param irrigated_fraction_dir: Input directory path for irrigated fraction data.
    :param rainfed_cropland_output_dir: Output directory path for classified rainfed cropland data.
    :param irrigated_cropland_output_dir: Output directory path for classified irrigated cropland data.
    :param interim_dir_for_maximum_occurrence_rainfed_cropland: interim directory path for classified rainfed cropland
                                                                data generated by "maximum occurrence approach", which
                                                                will later be finalized using a irrigated
                                                                cropland filter.
    :param skip_processing: Set to True if want to skip classifying irrigated and rainfed cropland data.

    :return: None
    """
    if not skip_processing:
        makedirs([rainfed_cropland_output_dir, irrigated_cropland_output_dir,
                  interim_dir_for_maximum_occurrence_rainfed_cropland])

        ############################
        # criteria of irrigated and rainfed cropland classification
        # more than 0.20 (20%) rainfed 30m pixels in a 2km pixel will be classified as "Rainfed cropland"
        # more than 0.20 (20%) rainfed 30m pixels in a 2km pixel will be classified as "Irrigated cropland"
        # These classifications may not be non-exclusive. This means a certain pixel might be dominantly irrigated but
        # have some (>20%) rainfed, in that case, that pixel will be classified as rainfed too. This helps with
        # training data extraction for effective precipitation model.

        rainfed_frac_threshold = 0.20
        irrigated_frac_threshold = 0.20

        # list of years when there are both irrigated and rainfed fraction datasets derived from
        # IrrMapper/LANID and USDA CDL. Classifying those data with defined threshold
        years_with_both_irrigated_rainfed_frac_data = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
                                                       2017, 2018, 2019, 2020]

        for year in years_with_both_irrigated_rainfed_frac_data:
            print(f'Classifying rainfed and irrigated cropland data for year {year}')

            rainfed_frac_data = os.path.join(rainfed_fraction_dir, f'Rainfed_Frac_{year}.tif')
            irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')

            rain_arr, rain_file = read_raster_arr_object(rainfed_frac_data)
            irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

            # classification using defined rainfed and irrigated fraction threshold. -9999 is no data
            rainfed_cropland = np.where(
                (rain_arr >= rainfed_frac_threshold) & ((irrig_arr < irrigated_frac_threshold) | (np.isnan(irrig_arr))),
                1, -9999)
            irrigated_cropland = np.where(irrig_arr >= irrigated_frac_threshold, 1, -9999)

            # saving classified data
            output_rainfed_cropland_raster = os.path.join(rainfed_cropland_output_dir, f'Rainfed_cropland_{year}.tif')
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir, f'Irrigated_cropland_{year}.tif')

            write_array_to_raster(raster_arr=rainfed_cropland, raster_file=rain_file, transform=rain_file.transform,
                                  output_path=output_rainfed_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case

        ############################
        # irrigated fraction data is also for 2000-2007. Classifying those data to irrigated cropland with defined threshold
        # Also, extending rainfed cropland data for 2000-2007 using "maximum occurrence approach"
        years_rest_irrigated_frac_data = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]

        for year in years_rest_irrigated_frac_data:
            print(f'Classifying rainfed and irrigated cropland data for year {year}')

            irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')

            irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

            # classification using defined rainfed and irrigated fraction threshold. -9999 is no data
            irrigated_cropland = np.where(irrig_arr >= irrigated_frac_threshold, 1, -9999)

            # saving classified data
            output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir, f'Irrigated_cropland_{year}.tif')
            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case

            # rainfed cropland fraction data is not available for 2000-2007 due to limited/no spatial coverage of USDA CDL
            # data. Applying maximum occurrence approach. It assigns rainfed cropland (value 1) for a pixel if that pixel
            # is rainfed in most years between 2008-2012 (user defined years).
            max_occurrence_rainfed_input_rasters = glob(os.path.join(rainfed_cropland_output_dir, '*200[8-9]*.tif')) + \
                                                   glob(os.path.join(rainfed_cropland_output_dir, '*201[0-2]*.tif'))

            max_occurrence_rainfed_raster = \
                apply_maximum_occurrence_approach(input_rasters_list=max_occurrence_rainfed_input_rasters,
                                                  output_dir=interim_dir_for_maximum_occurrence_rainfed_cropland,
                                                  raster_name=f'Rainfed_cropland_{year}.tif')

            # applying an additional filter of irrigated cropland (we already classified irrigated cropland for 2000-2007).
            # this filter will remove any pixel classified as rainfed from "maximum occurrence approach" if it has already
            # been classified as irrigated cropland (we classified irrigated cropland data from IrrMapper/LANID-derived
            # irrigated fraction data; therefore, have more confidence on these dataset)
            max_occur_arr, max_occur_file = read_raster_arr_object(max_occurrence_rainfed_raster)
            filtered_rainfed_arr = np.where((max_occur_arr == 1) & (irrigated_cropland != 1), 1, -9999)

            final_rainfed_raster = os.path.join(rainfed_cropland_output_dir, f'Rainfed_cropland_{year}.tif')
            write_array_to_raster(raster_arr=filtered_rainfed_arr, raster_file=max_occur_file,
                                  transform=max_occur_file.transform, output_path=final_rainfed_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case

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
        # cropET datasets have been extracted from openET for the following years and months only
        years_to_filter_cropET = [2016, 2017, 2018, 2019, 2020]
        months_to_filter_cropET = list(range(1, 13))

        for year in years_to_filter_cropET:
            print(f'Filtering cropET data for year {year}...')

            # pure rainfed cropland filtered previously by using rainfed and irrigated fraction threshold
            # (rainfed frac > 0.40 and irrig frac < 0.10)
            rainfed_cropland_data = glob(os.path.join(rainfed_cropland_dir, f'*{year}*.tif'))[0]
            rainfed_cropland_arr = read_raster_arr_object(rainfed_cropland_data, get_file=False)

            # pure irrigated cropland filtered previously by using irrigated fraction threshold (irrig frac > 0.10)
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)

            for month in months_to_filter_cropET:
                # # applying rainfed cropland filter to get cropET at purely rainfed pixels
                rainfed_cropET_data = glob(os.path.join(rainfed_cropET_input_dir, f'*{year}_{month}*.tif'))[0]
                rainfed_cropET_arr, rainfed_cropET_file = read_raster_arr_object(rainfed_cropET_data)

                # applying the filter
                rainfed_cropET_arr[np.isnan(rainfed_cropland_arr)] = -9999

                filtered_output_raster = os.path.join(rainfed_cropET_output_dir, f'Rainfed_cropET_{year}_{month}.tif')
                write_array_to_raster(raster_arr=rainfed_cropET_arr, raster_file=rainfed_cropET_file,
                                      transform=rainfed_cropET_file.transform, output_path=filtered_output_raster)

                # # applying irrigated cropland filter to get cropET at purely irrigated pixels
                irrigated_cropET_data = glob(os.path.join(irrigated_cropET_input_dir, f'*{year}_{month}*.tif'))[0]
                irrigated_cropET_arr, irrigated_cropET_file = read_raster_arr_object(irrigated_cropET_data)

                # applying the filter
                irrigated_cropET_arr[np.isnan(irrigated_cropland_arr)] = -9999

                filtered_output_raster = os.path.join(irrigated_cropET_output_dir, f'Irrigated_cropET_{year}_{month}.tif')
                write_array_to_raster(raster_arr=irrigated_cropET_arr, raster_file=irrigated_cropET_file,
                                      transform=irrigated_cropET_file.transform, output_path=filtered_output_raster)

    else:
        pass


def process_ssebop_data(input_ssebop_dir, output_dir_ssebop_monthly, output_dir_ssebop_yearly,
                        years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                               2014, 2015, 2016, 2017, 2018, 2019, 2020),
                        west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                        ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) Ssebop ET.

    :param input_ssebop_dir: Directory file path of downloaded ssebop datasets.
    :param output_dir_ssebop_monthly: File path of directory to save monthly SseBop ET at Western US extent.
    :param output_dir_ssebop_yearly: File path of directory to save summed ET for each year at Western US extent.
    :param years: Tuple/list of years for which ssebop data was downloaded.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip ssebop processing.

    :return: None.
    """
    if not skip_processing:
        interim_dir_for_clip = os.path.join(input_ssebop_dir, 'interim_dir_clip')
        interim_dir_for_sum = os.path.join(input_ssebop_dir, 'interim_dir_sum')
        makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_ssebop_monthly, output_dir_ssebop_yearly])

        #########
        # # Code-block for saving monthly data for the Western US
        #########
        # Clipping SseBop monthly datasets for Western US
        monthly_ssebop_data = glob(os.path.join(input_ssebop_dir, '*.tif'))  # monthly ssebop datasets
        for data in monthly_ssebop_data:
            month = os.path.basename(data).split('.')[0][-2:]
            year = os.path.basename(data).split('.')[0][1:5]

            if month.startswith('0'):  # don't want to keep 0 in month for consistency will all datasets
                month = month[-1]

            print(f'processing SseBop data for year {year}....')

            monthly_raster_name = f'SSEBOP_ET_{year}_{month}.tif'

            clip_resample_reproject_raster(input_raster=data,
                                           input_shape=west_US_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_ssebop_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=True, ref_raster=ref_raster,
                                           resolution=resolution)

        #########
        # # Code-block for summing monthly data for years by growing season for the Western US
        #########
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
            print(f'summing Ssebop data for year {year}...')
            ssebop_datasets = glob(
                os.path.join(input_ssebop_dir, f'*{year}*.tif'))  # monthly ssebop datasets for each year

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
            _, mosaiced_ssebop = mosaic_rasters_from_directory(input_dir=interim_dir_for_sum, output_dir=output_dir_ssebop_yearly,
                                                               raster_name=yearly_summed_ssebop, ref_raster=ref_raster,
                                                               search_by=search_by, resolution=resolution)
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


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, output_dir_prism_yearly,
                       years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       keyword='prism_precip',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) Prism Precipitation.

    :param prism_bil_dir: Directory file path of downloaded prism datasets in .bil format.
    :param prism_tif_dir: Directory file path of prism datasets converted to tif format.
    :param output_dir_prism_monthly: File path of directory to save monthly prism precipitation/temperature data for
                                     at Western US extent.
    :param output_dir_prism_yearly: File path of directory to save summed/mean prism precipitation/temperature data for
                                    each year at Western US extent.
    :param years: Tuple/list of years for which prism data was downloaded.
    :param keyword: keyword to add before processed datasets. Can take 'prism_precip', ''prism_tmax', 'prism_tmin'.
                    Default set to 'prism_precip'.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: None.
    """
    if not skip_processing:
        convert_prism_data_to_tif(input_dir=prism_bil_dir, output_dir=prism_tif_dir, keyword=keyword)

        interim_dir_for_clip = os.path.join(prism_tif_dir, 'interim_dir_clip')

        # Creating additional folders/directories needed for processing (clip/sum/mean)
        if 'precip' in keyword:
            interim_dir_for_sum = os.path.join(prism_tif_dir, 'interim_dir_sum')
            makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_prism_yearly])

        elif any(i in keyword for i in ['tmax', 'tmin']):
            interim_dir_for_mean = os.path.join(prism_tif_dir, 'interim_dir_mean')
            makedirs([interim_dir_for_clip, interim_dir_for_mean, output_dir_prism_monthly, output_dir_prism_yearly])

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

            clip_resample_reproject_raster(input_raster=data,
                                           input_shape=west_US_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_prism_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=True, ref_raster=ref_raster,
                                           resolution=resolution)

        #########
        # # Code-block for summing monthly data for years by growing season for the Western US
        #########
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
            prism_datasets = glob(os.path.join(prism_tif_dir, f'*{year}*.tif'))  # monthly prism datasets for each year

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
                    _, mosaiced_prism = mosaic_rasters_from_directory(input_dir=interim_dir_for_sum,
                                                                      output_dir=output_dir_prism_yearly,
                                                                      raster_name=yearly_prism, ref_raster=ref_raster,
                                                                      search_by=search_by, resolution=resolution)

                elif any(i in keyword for i in ['tmax', 'tmin']):
                    # Calculating mean of raster for each growing season
                    mean_output_for_season = os.path.join(interim_dir_for_mean, f'prism_{year}_' + season + '.tif')
                    mean_rasters(raster_list=prism_clipped_list, raster_dir=None, output_raster=mean_output_for_season,
                                 ref_raster=prism_clipped_list[0])

                    # mosaicing prism temperature for all growing seasons in a year
                    yearly_prism = f'{keyword}_{year}.tif'
                    search_by = f'*{year}*.tif'
                    _, mosaiced_prism = mosaic_rasters_from_directory(input_dir=interim_dir_for_mean,
                                                                      output_dir=output_dir_prism_yearly,
                                                                      raster_name=yearly_prism, ref_raster=ref_raster,
                                                                      search_by=search_by, resolution=resolution)
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


def run_all_preprocessing(skip_ssebop_processing=False,
                          skip_prism_processing=False,
                          skip_merging_rainfed_frac=False,
                          skip_merging_rainfed_cropET=False,
                          skip_merging_irrigated_frac=False,
                          skip_merging_irrigated_cropET=False,
                          skip_classifying_irrigated_rainfed_cropland=False,
                          skip_filtering_irrigated_rainfed_cropET=False,
                          skip_processing_slope_data=False,
                          ref_raster=WestUS_raster):
    """
    Run all preprocessing steps.

    :param skip_processing_slope_data:
    :param skip_ssebop_processing: Set True if want to skip ssebop data preprocessing.
    :param skip_prism_processing: Set True if want to skip prism (precipitation and temperature) data preprocessing.
    :param skip_merging_rainfed_frac: Set to True to skip merging rainfed fraction data.
    :param skip_merging_rainfed_cropET: Set to True to skip merging rainfed cropET data.
    :param skip_merging_irrigated_frac: Set to True to skip merging irrigated fraction data.
    :param skip_merging_irrigated_cropET: Set to True to skip merging irrigated fraction data.
    :param skip_classifying_irrigated_rainfed_cropland: Set to True if want to skip classifying irrigated and
                                                        rainfed cropland data.
    :param skip_filtering_irrigated_rainfed_cropET: Set to True if want to skip filtering irrigated and rainfed cropET
                                                    data.
    :param ref_raster: Filepath of Western US reference raster to use in 2km pixel lat-lon raster creation and to use
                        as reference raster in other processing operations.

    :return: None.
    """
    # merge rainfed fraction dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(input_dir_irrmapper='../../Data_main/Raster_data/Rainfed_Frac_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Rainfed_Frac_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac',
                                                   merge_keyword='Rainfed_Frac', monthly_data=False,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_rainfed_frac)

    # merge rainfed cropET dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(input_dir_irrmapper='../../Data_main/Raster_data/Rainfed_crop_OpenET_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Rainfed_crop_OpenET_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly_raw',
                                                   merge_keyword='Rainfed_cropET', monthly_data=True,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_rainfed_cropET)

    # merge irrigated fraction dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(input_dir_irrmapper='../../Data_main/Raster_data/Irrigation_Frac_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Irrigation_Frac_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
                                                   merge_keyword='Irrigated_Frac', monthly_data=False,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_irrigated_frac)

    # merge irrigated cropET dataset
    merge_GEE_data_patches_IrrMapper_LANID_extents(input_dir_irrmapper='../../Data_main/Raster_data/Irrig_crop_OpenET_IrrMapper',
                                                   input_dir_lanid='../../Data_main/Raster_data/Irrig_crop_OpenET_LANID',
                                                   merged_output_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly_raw',
                                                   merge_keyword='Irrigated_cropET', monthly_data=True,
                                                   ref_raster=WestUS_raster,
                                                   skip_processing=skip_merging_irrigated_cropET)

    # classify rainfed and irrigated cropland data
    classify_irrigated_rainfed_cropland(
        rainfed_fraction_dir='../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac',
        irrigated_fraction_dir='../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
        rainfed_cropland_output_dir='../../Data_main/Raster_data/Rainfed_cropland',
        irrigated_cropland_output_dir='../../Data_main/Raster_data/Irrigated_cropland',
        interim_dir_for_maximum_occurrence_rainfed_cropland='../../Data_main/Raster_data/Rainfed_cropland/interim_dir_for_max_occurrence',
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

    # ssebop ET data processing
    process_ssebop_data(input_ssebop_dir='../../Data_main/Raster_data/Ssebop_ETa/raw_data',
                        output_dir_ssebop_monthly='../../Data_main/Raster_data/Ssebop_ETa/WestUS_monthly',
                        output_dir_ssebop_yearly='../../Data_main/Raster_data/Ssebop_ETa/WestUS',
                        years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                               2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                        ref_raster=WestUS_raster,
                        skip_processing=skip_ssebop_processing)

    # prism precipitation data processing
    process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/Raster_data/PRISM_Precip/bil_format',
                       prism_tif_dir='../../Data_main/Raster_data/PRISM_Precip/tif_format',
                       output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly',
                       output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_Precip/WestUS',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_precip', skip_processing=skip_prism_processing)

    # prism maximum temperature data processing
    process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/Raster_data/PRISM_Tmax/bil_format',
                       prism_tif_dir='../../Data_main/Raster_data/PRISM_Tmax/tif_format',
                       output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_Tmax/WestUS_monthly',
                       output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_Tmax/WestUS',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_tmax', skip_processing=skip_prism_processing)

    # prism minimum temperature data processing
    process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/Raster_data/PRISM_Tmin/bil_format',
                       prism_tif_dir='../../Data_main/Raster_data/PRISM_Tmin/tif_format',
                       output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_Tmin/WestUS_monthly',
                       output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_Tmin/WestUS',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_tmin', skip_processing=skip_prism_processing)

    # converting DEM data to slope
    create_slope_raster(input_raster='../../Data_main/Raster_data/DEM/WestUS/DEM.tif',
                        output_dir='../../Data_main/Raster_data/Slope', raster_name='Slope.tif',
                        skip_processing=skip_processing_slope_data)

    # making a latitude longitude raster from reference raster
    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    lon_arr, lat_arr = make_lat_lon_array_from_raster(ref_raster)

    lon_dir = os.path.join('../../Data_main/Raster_data', 'Longitude')
    lat_dir = os.path.join('../../Data_main/Raster_data', 'Latitude')
    makedirs([lon_dir, lat_dir])

    write_array_to_raster(raster_arr=lon_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=os.path.join(lon_dir, 'Longitude.tif'))
    write_array_to_raster(raster_arr=lat_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=os.path.join(lat_dir, 'Latitude.tif'))




