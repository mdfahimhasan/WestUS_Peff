import os
import pickle
import numpy as np
from glob import glob
from osgeo import gdal

from Codes.utils.system_ops import makedirs, copy_file
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, mosaic_rasters, \
    clip_resample_reproject_raster, mask_raster_by_extent, sum_rasters, mean_rasters, filter_raster_on_threshold


no_data_value = -9999
model_res = 0.02000000000000000736  # in deg, ~2.22 km
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def apply_maximum_occurrence_approach(input_rasters_list, output_dir, raster_name):
    """
    **** Best works for binary rasters ****
    Creates a output binary raster using maximum occurrence approach (a pixel in 1 if most of the input raster
    shows 1 in that pixel) incorporating the input rasters.

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
        input_raster_dict[name] = read_raster_arr_object(data, get_file=False, change_dtype=False).flatten()

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
    print(max_arr.dtype)
    # converting the data type to Float32
    max_arr = max_arr.astype(np.float32)
    print(max_arr.dtype)
    # replacing the nan pixels with -9999 value
    max_arr = np.where(~np.isnan(ref_arr), max_arr, ref_arr)

    # Writing result of maximum occurrence
    makedirs([output_dir])
    output_raster = os.path.join(output_dir, raster_name)
    write_array_to_raster(raster_arr=max_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)

    return output_raster


def process_cdl_data_to_cropland(cdl_dir='../../Data_main/Raster_data/USDA_CDL/CONUS_original_30m',
                                 processing_shp_dir='../../Data_main/Raster_data/USDA_CDL/processing_shapes',
                                 westus_30m_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_30m',
                                 westus_processed_cropland_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_Processed_cropland',
                                 ref_raster=WestUS_raster, resolution=model_res,
                                 already_masked=False, try_maximum_occurrence_approach=False,
                                 skip_processing=False):
    """
    Mask and process cdl datasets to only cropland. Processing includes selecting only cropland pixels and resampling
    to 2km resolution.

    CDL data download link : https://www.nass.usda.gov/Research_and_Science/Cropland/Release/

    :param cdl_dir: Directory path of original (CONUS scale) CDL datasets.
    :param processing_shp_dir: Directory path of shapefiles that is used to process cdl data. cdl datasets are 30m
                               resolution and memory-heavy. These shapefiles are used to mask the cdl datasets,
                               which is later processed individually.
    :param westus_30m_dir: Output directory path for saving Western US scale 30m cdl data.
                           Also, automatically saves intermediate datasets before final cropland products.
    :param westus_processed_cropland_dir: Output directory path for saving finalized Western US scale 2km cropland data.

    :param ref_raster: Filepath of Western US reference raster (set to default).
    :param resolution: Model resolution. Default set to 0.02000000000000000736.
    :param already_masked: Set to True to skip masking (masked cdl data has already been prepared).
    :param try_maximum_occurrence_approach: Defaults to False to set 2008 dataset value as 2000 & 2005 datasets.
                                            'try_maximum_occurrence_approach': (set to True to use)
                                            2000 and 2005 cdl datasets for whole Western US isn't available.
                                            This approach sets the maximum occurring cropland/non-cropland values from
                                            2008-2011 as 2000 & 2005 datasets.
    :param skip_processing: Set to True to skip cdl cropland dataset processing.

    :return: None.
    """
    if not skip_processing:
        print('processing cdl data to cropland...')
        cdl_US = glob(os.path.join(cdl_dir, '*.tif'))
        processing_bounds = glob(os.path.join(processing_shp_dir, '*.shp'))

        cdl_2008_arr, cdl_2009_arr, cdl_2010_arr, cdl_2011_arr, cdl_file, shape = None, None, None, None, None, None
        if not already_masked:
            # Loop for masking cdl raster to Western US extent.
            # Using 4 bounding boxes to mask and split the cdl for Western US into 4 regions for each year
            # to minimize memory load.
            for cdl_ras in cdl_US:
                for shape in processing_bounds:
                    raster_name = os.path.splitext(os.path.basename(cdl_ras))[0]
                    shape_name = os.path.splitext(os.path.basename(shape))[0]
                    name = raster_name + '_' + shape_name + '.tif'
                    mask_raster_by_extent(input_raster=cdl_ras, ref_file=shape, output_dir=westus_30m_dir,
                                          raster_name=name, invert=False, crop=True, nodata=0)

        # Creating new directory inside westus_30m_dir to save processed cropland data
        # Creating new directory to save processed cropland data
        cropland_30m_dir = os.path.join(westus_30m_dir, 'cropland')
        makedirs([cropland_30m_dir, westus_processed_cropland_dir])

        # converting to crop vs non-crop and developed vs non-developed data
        noncrop_classes = [0, 58, 59, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 121, 122, 123, 124, 131, 141, 142,
                           143, 152, 176, 190, 195]  # 0 is no data value

        # Listing all 30m cdl datasets for WestUS (each year consists of 4 cdl datasets, each for a particular region)
        cdl_westus = glob(os.path.join(westus_30m_dir, '*.tif'))

        # cropped each year's cdl data in 4 shapefile bounds for memory issue. They are specified as 'bound1', 'bound2',...
        # at the end of each raster's name. In this tep, we will select only the cropland pixels, resample them to 2km.
        for cdl_ras in cdl_westus:
            # crop vs non-crop raster  # if array value in noncrop list assigns 0, otherwise assigns 1
            # cdl data has no_data value set to 0
            cdl_arr, cdl_file = read_raster_arr_object(cdl_ras, change_dtype=False)  # datatype unsigned integer 8bit
            crop_arr = np.where(~np.isin(cdl_arr, noncrop_classes), 1, 0)

            crop_raster_name = os.path.splitext(os.path.basename(cdl_ras))[0] + '_crop.tif'
            crop_raster_path = os.path.join(cropland_30m_dir, crop_raster_name)

            # saving 30m cropland rasters
            # setting nodata to None, otherwise all 0 values will become nodata.
            # nodata pixels will be assigned to -9999 at mosaicing step
            crop_30m_raster = write_array_to_raster(raster_arr=crop_arr, raster_file=cdl_file,
                                                    transform=cdl_file.transform, output_path=crop_raster_path,
                                                    nodata=None, dtype=np.uint8)

            # Resampling 30m raster to 2km using 'average' resampling algorithm
            # the values represent percent (weighted average) cropland in the pixel
            year = crop_raster_name.split('_')[0]
            bound_no = crop_raster_name.split('_')[3]
            resampled_raster_name = f'{year}_2km_crop_percent_{bound_no}.tif'
            resampled_raster = clip_resample_reproject_raster(input_raster=crop_30m_raster, input_shape=None,
                                                              raster_name=resampled_raster_name,
                                                              output_raster_dir=cropland_30m_dir,
                                                              clip=False, resample=True, clip_and_resample=False,
                                                              targetaligned=True, resample_algorithm='average',
                                                              resolution=resolution, crs='EPSG:4269',
                                                              output_datatype=gdal.GDT_Float32)

            # filtering resampled rasters by a value threshold and setting them as cropland (value=1)
            output_raster = os.path.join(cropland_30m_dir, f'{year}_2km_{bound_no}.tif')
            filter_raster_on_threshold(input_raster=resampled_raster, output_raster=output_raster,
                                       threshold_value1=0.15,
                                       assign_value=1, refraster=resampled_raster)

        # In this step, we are mosaicin each year's datasets to form a single cropland dataset for each year
        year_list = [2008, 2009, 2010, 2011, 2015]

        for year in year_list:
            search_by = f'*{year}_2km_bound*.tif'
            raster_name = f'USDA_cropland_{year}.tif'
            mosaic_rasters(input_dir=cropland_30m_dir, output_dir=westus_processed_cropland_dir,
                           raster_name=raster_name, ref_raster=ref_raster, search_by=search_by,
                           dtype=np.float32, mosaicing_method='max', resolution=resolution, nodata=no_data_value)
            print(f'cdl data for {year} processed to cropland')

        # 2000 and 2005 cdl data isn't available for all Western US.
        # 'try_maximum_occurrence_approach' compares each pixel between 2008-2011.
        # Assigns cropland (value 1) for a pixel if that pixel is cropland in most years between 2008-2011
        if try_maximum_occurrence_approach:
            # Reading 2008-2011 cropland data in memory to apply maximum occurence approach
            westus_cdl_crops = glob(os.path.join(westus_processed_cropland_dir, '*.tif'))

            for crop_cdl in westus_cdl_crops:
                if '2008' in crop_cdl:
                    cdl_2008_arr, cdl_file = read_raster_arr_object(crop_cdl)
                    shape = cdl_2008_arr.shape
                    cdl_2008_arr = cdl_2008_arr.flatten()
                elif '2009' in crop_cdl:
                    cdl_2009_arr = read_raster_arr_object(crop_cdl, get_file=False).flatten()
                elif '2010' in crop_cdl:
                    cdl_2010_arr = read_raster_arr_object(crop_cdl, get_file=False).flatten()
                elif '2011' in crop_cdl:
                    cdl_2011_arr = read_raster_arr_object(crop_cdl, get_file=False).flatten()

            # stacking 2008-2011 cropland cdl arrays
            max_arr = np.stack([cdl_2008_arr, cdl_2009_arr, cdl_2010_arr, cdl_2011_arr], axis=0)
            # creating an empty array to assign cropland (1) if most years between 2008-2011 are cropland
            new_arr = np.zeros(cdl_2008_arr.shape)

            # this loop takes each column of stacked array. np.bincount() counts the number of occurence of each value
            # in the selected column. Then, np.argmax() selects the value which occurs most frequently. Then, we paste
            # the selected value to a new array
            for i in range(max_arr.shape[1]):
                selected_arr = max_arr[:, i]
                selected_arr = np.where(np.isnan(selected_arr), 0, selected_arr)  # replacing Nan values with 0

                max_occurred_value = np.argmax(np.bincount(list(selected_arr)))  # # # improve this. try to avoid loop
                new_arr[i] = max_occurred_value

            max_arr = new_arr.reshape(shape)

            # replacing the nan pixels with -9999 value
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            max_arr = np.where(~np.isnan(ref_arr), max_arr, ref_arr)

        else:  # if try_maximum_occurrence_approach=False, assigns 2008 cropland arrays
            # value as 2000 and 2005 cropland data
            cdl_2008_data = glob(os.path.join(westus_processed_cropland_dir, '*2008.tif'))
            cdl_2008_arr, cdl_file = read_raster_arr_object(cdl_2008_data)
            max_arr = cdl_2008_arr.reshape(cdl_2008_arr.shape)

        # writing generated cropland rasters for 2000 and 2005
        output_2000_cdl_data = os.path.join(westus_processed_cropland_dir, 'USDA_cropland_2000.tif')
        output_2005_cdl_data = os.path.join(westus_processed_cropland_dir, 'USDA_cropland_2005.tif')

        write_array_to_raster(max_arr, raster_file=cdl_file, transform=cdl_file.transform,
                              output_path=output_2000_cdl_data)
        write_array_to_raster(max_arr, raster_file=cdl_file, transform=cdl_file.transform,
                              output_path=output_2005_cdl_data)
        print('cdl cropland data for 2000 and 2005 generated')

        # Copying processed cdl cropland data to compiled data folder and saving in a dictionary to load later
        westus_cdl_crops = glob(os.path.join(westus_processed_cropland_dir, '*.tif'))
        cdl_cropland_processing_dict = {}
        for crop_cdl in westus_cdl_crops:
            data_name = os.path.basename(crop_cdl)
            copied_data = copy_file(crop_cdl, copy_dir='../../Data_main/Compiled_data')
            cdl_cropland_processing_dict[data_name] = copied_data
            pickle.dump(cdl_cropland_processing_dict, open('../../Data_main/Compiled_data/cdl_cropland_dict.pkl',
                                                           mode='wb+'))
    else:
        cdl_cropland_processing_dict = pickle.load(open('../../Data_main/Compiled_data/cdl_cropland_dict.pkl',
                                                        mode='rb'))

    return cdl_cropland_processing_dict


def process_irrigated_LANID_data(input_dir='../../Data_main/Raster_data/LANID',
                                 input_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                                 skip_processing=False,
                                 ref_raster=WestUS_raster, resampling_resolution=model_res,
                                 try_maximum_occurrence_approach=True):
    """
    Process irrigated cropland data (LANID) from Xie et al. 2021. Two sets of rasters are generated.
    First: Irrigated area raster with values 0 and 1 (1 is irrigated);
    second: Irrigated fraction data (values in fraction from 0 or 1, higher fraction means higher
    percentage of irrigated lands in that 2-km pixel).

    :param input_dir: Filepath of input data directory.
    :param input_shape: Filepath of ROI shapefile.
    :param skip_processing: Bool. Set to True is want to skip processing.
    :param ref_raster: Filepath of Western US reference raster.
    :param resampling_resolution: Model resolution used for resampling dataset.
                                  Here, the model resolution is only used for resampling CONUS-scale dataset.
                                  At later step of masking/clipping, output raster dimension was used from the
                                  reference raster.
    :param try_maximum_occurrence_approach: Defaults to False to set 2017 dataset value as 2018-2022 datasets.
                                            'try_maximum_occurrence_approach': (set to true to use)
                                            2018-2022 irrigated cropland data aren't available.
                                            'try_maximum_occurrence_approach' compares each pixel between 2013-2017.
                                            Assigns irrigated cropland (value 1) for a pixel if that pixel is irrigated
                                            in most years between 2008-2011.

    :return: A dictionary of processed dataset filepaths. The dictionary has years (e.g., 2015) as keys.
    """
    irrigated_land_dict = None

    if not skip_processing:
        print('Processing Irrigated Agriculture dataset....')
        input_raster = glob(os.path.join(input_dir, '*.tif'))

        irrigation_land_dict = {}  # a dictionary to store irrigated area data (values in 0 or 1, 1 means irrigated)

        # creating output directory
        interim_dir = os.path.join(input_dir, 'interim_data')
        irrigated_frac_dir = os.path.join(input_dir, 'WestUS_Processed_irrigated_frac')
        irrigated_land_dir = os.path.join(input_dir, 'WestUS_Processed_irrigated_lands')
        makedirs([interim_dir, irrigated_frac_dir, irrigated_land_dir])

        for raster in input_raster:
            if any([i in raster for i in ['clipped', 'resample', 'irrigated']]):
                os.remove(raster)  # to remove raster previously processed
            else:
                # The dataset is 30m resolution CONUS scale and huge to clip and resample at the same time
                # Resampling to model resolution first and then clipping. "average" resampling calculates the
                # fraction of irrigated area (30m pixel) in each 2-km pixel.

                # # Processing irrigated area data to fraction of irrigated area data
                # resampling (by weighted average method) CONUS-scale dataset to 2km
                # Using model resolution for resampling
                resampled_raster = clip_resample_reproject_raster(
                    input_raster=raster, input_shape=input_shape, keyword='resampled', resolution=resampling_resolution,
                    output_raster_dir=interim_dir, clip_and_resample=False, clip=False, resample=True,
                    resample_algorithm='average', use_ref_width_height=False)

                # Masking resampled dataset to Western US scale
                # Using width height of reference raster. 'resolution' set to None.
                year = os.path.splitext(os.path.basename(resampled_raster))[0][-4:]
                output_name = f'irrigated_agri_frac_{year}'
                clipped_raster = clip_resample_reproject_raster(
                    input_raster=resampled_raster, input_shape=input_shape, keyword=' ', raster_name=output_name,
                    resolution=None, output_raster_dir=irrigated_frac_dir,
                    clip_and_resample=True, clip=False, resample=False, targetaligned=False,
                    use_ref_width_height=True,
                    ref_raster=ref_raster)  # targetAlignedPixels=False to ensure equal row-columns like other dataset

                # # Processing irrigated area data to presence of irrigated area data
                # Using threshold to choose irrigated area
                output_name = f'irrigated_agri_{year}'
                output_fp = os.path.join(irrigated_land_dir, f'{output_name}.tif')
                filter_raster_on_threshold(input_raster=clipped_raster, output_raster=output_fp, threshold_value1=0.15,
                                           assign_value=1)

                # copying to compiled directory
                copied_data = copy_file(input_dir_file=output_fp, copy_dir='../../Data_main/Compiled_data', rename=None)
                irrigation_land_dict[year] = copied_data

        # 2018-2022 irrigated cropland data aren't available.
        # 'try_maximum_occurrence_approach' compares each pixel between 2013-2017.
        # Assigns irrigated cropland (value 1) for a pixel if that pixel is irrigated
        # in most years between 2008-2011
        irrig_2013_arr, irrig_2014_arr, irrig_2015_arr, irrig_2016_arr, irrig_2017_arr = None, None, None, \
                                                                                         None, None
        if try_maximum_occurrence_approach:
            # Reading 2013-2017 cropland data in memory to apply maximum occurrence approach
            westus_irrigated_crops = glob(os.path.join(irrigated_land_dir, '*201[3-7].tif'))
            print(westus_irrigated_crops)

            for irrigated_crop in westus_irrigated_crops:
                if '2013' in irrigated_crop:
                    irrig_2013_arr, irrig_file = read_raster_arr_object(irrigated_crop)
                    shape = irrig_2013_arr.shape
                    irrig_2013_arr = irrig_2013_arr.flatten()
                elif '2014' in irrigated_crop:
                    irrig_2014_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()
                elif '2015' in irrigated_crop:
                    irrig_2015_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()
                elif '2016' in irrigated_crop:
                    irrig_2016_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()
                elif '2017' in irrigated_crop:
                    irrig_2017_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()

            # stacking 2013-2017 irrigated cropland arrays
            max_arr = np.stack([irrig_2013_arr, irrig_2014_arr, irrig_2015_arr, irrig_2016_arr, irrig_2017_arr],
                               axis=0)
            # creating an empty array to assign cropland (1) if most years between 2008-2011 are cropland
            new_arr = np.zeros(irrig_2017_arr.shape)

            # this loop takes each column of stacked array. np.bincount() counts the number of occurence of
            # each value in the selected column. Then, np.argmax() selects the value which occurs most
            # frequently. Then, we paste the selected value to a new array
            for i in range(max_arr.shape[1]):
                selected_arr = max_arr[:, i]
                selected_arr = np.where(np.isnan(selected_arr), 0, selected_arr)  # replacing Nan values with 0

                max_occurred_value = np.argmax(
                    np.bincount(list(selected_arr)))
                new_arr[i] = max_occurred_value

            max_arr = new_arr.reshape(shape)

            # replacing the nan pixels with -9999 value
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            max_arr = np.where(~np.isnan(ref_arr), max_arr, ref_arr)

        else:  # if try_maximum_occurrence_approach=False, assigns 2017 irrigated cropland arrays value
            # as 2018-2022 irrigated cropland data
            irrig_crop_2017_data = glob(os.path.join(irrigated_land_dir, '*2017.tif'))
            irrig_2017_arr, irrig_file = read_raster_arr_object(irrig_crop_2017_data)
            max_arr = irrig_2017_arr.reshape(irrig_2017_arr.shape)

        # writing generated cropland rasters for 2018-2022
        output_2018_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2018.tif')
        output_2019_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2019.tif')
        output_2020_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2020.tif')
        output_2021_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2021.tif')
        output_2022_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2022.tif')

        write_copy_files = [output_2018_irrigated_data, output_2019_irrigated_data, output_2020_irrigated_data,
                            output_2021_irrigated_data, output_2022_irrigated_data]

        for each_data in write_copy_files:
            # writing created file
            write_array_to_raster(max_arr, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=each_data)
            # copying to compiled directory
            copied_data = copy_file(input_dir_file=each_data, copy_dir='../../Data_main/Compiled_data', rename=None)
            year = os.path.splitext(os.path.basename(each_data))[0][-4:]
            irrigation_land_dict[year] = copied_data

            print('Irrigated cropland data for 2018-2022 generated')

        print('Processed Irrigated Agriculture dataset')
        pickle.dump(irrigation_land_dict, open('../../Data_main/Compiled_data/irrigated_land_dict.pkl', mode='wb+'))

    else:
        irrigated_land_dict = pickle.load(open('../../Data_main/Compiled_data/irrigated_land_dict.pkl', mode='rb'))

    return irrigated_land_dict


def process_ssebop_data(input_ssebop_dir, output_dir_ssebop_monthly, output_dir_ssebop_yearly,
                        years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                               2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
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

    :return: A dictionary with ssebop data file paths.
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
            print(f'Processing Ssebop data for {year}...')
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
            _, mosaiced_ssebop = mosaic_rasters(input_dir=interim_dir_for_sum, output_dir=output_dir_ssebop_yearly,
                                                raster_name=yearly_summed_ssebop, ref_raster=ref_raster,
                                                search_by=search_by, resolution=resolution)

            # Copying mosaiced ssebop ET to compiled data folder and saving in a dictionary to load later
            copied_data = copy_file(mosaiced_ssebop, copy_dir='../../Data_main/Compiled_data')
            ssebop_processing_dict[yearly_summed_ssebop[:yearly_summed_ssebop.rfind('.')]] = copied_data
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


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, output_dir_prism_yearly,
                       years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
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
    :param keyword: keyword to add before processed datasets.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: A dictionary with prism precip data file paths.
    """
    if not skip_processing:
        convert_prism_data_to_tif(input_dir=prism_bil_dir, output_dir=prism_tif_dir, keyword=keyword)

        interim_dir_for_clip = os.path.join(prism_tif_dir, 'interim_dir_clip')

        # Creating additional folders/directories needed for processing (clip/sum/mean)
        if 'precip' in keyword:
            interim_dir_for_sum = os.path.join(prism_tif_dir, 'interim_dir_sum')
            makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_prism_yearly])
        elif 'temp' in keyword:
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
            if 'precip' in keyword:
                monthly_raster_name = f'prism_precip_{year}_{month}.tif'
            else:
                monthly_raster_name = f'prism_temp_{year}_{month}.tif'

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
                    _, mosaiced_prism = mosaic_rasters(input_dir=interim_dir_for_sum,
                                                       output_dir=output_dir_prism_yearly,
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
                    _, mosaiced_prism = mosaic_rasters(input_dir=interim_dir_for_mean,
                                                       output_dir=output_dir_prism_yearly,
                                                       raster_name=yearly_prism, ref_raster=ref_raster,
                                                       search_by=search_by, resolution=resolution)

            # Copying mosaiced prism precip to compiled data folder and saving in a dictionary to load later
            copied_data = copy_file(mosaiced_prism, copy_dir='../../Data_main/Compiled_data')
            prism_processing_dict[yearly_prism[:yearly_prism.rfind('.')]] = copied_data
            pickle.dump(prism_processing_dict, open('../../Data_main/Compiled_data/prism_dict.pkl', mode='wb+'))
    else:
        prism_processing_dict = pickle.load(open('../../Data_main/Compiled_data/prism_dict.pkl', mode='rb'))

        return prism_processing_dict

def run_all_preprocessing(skip_cdl_cropland_processing=False, cdl_already_masked=False,
                          try_maximum_occurrence_approach_for_cdl=True,
                          skip_ssebop_processing=False,
                          skip_prism_processing=False):
    """
    Run all preprocessing steps.

    :param skip_cdl_cropland_processing: Set to True to skip cdl cropland dataset processing.
    :param cdl_already_masked: Set to True to skip masking (if masked cdl data has already been prepared).
    :param try_maximum_occurrence_approach_for_cdl: Defaults to False to set 2008 dataset value as 2000 & 2005 datasets.
                                            'try_maximum_occurrence_approach': (set to True to use)
                                            2000 and 2005 cdl datasets for whole Western US isn't available.
                                            This approach sets the maximum occurring cropland/non-cropland values from
                                            2008-2011 as 2000 & 2005 datasets.
    :param skip_ssebop_processing: Set True if want to skip ssebop data preprocessing.
    :param skip_irrigatedCrop_processing: Set True if want to skip Irrigated cropland data preprocessing.
    :param skip_prism_processing: Set True if want to skip prism (precipitation and temperature) data preprocessing.

    :return: Dictionaries of crop and developed land use data file paths
    """
    # USDA cdl to cropland processing
    cdl_cropland_dict = \
        process_cdl_data_to_cropland(cdl_dir='../../Data_main/Raster_data/USDA_CDL/CONUS_original_30m',
                                     processing_shp_dir='../../Data_main/Raster_data/USDA_CDL/processing_shapes',
                                     westus_30m_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_30m',
                                     westus_processed_cropland_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_Processed_cropland',
                                     ref_raster=WestUS_raster, resolution=model_res,
                                     already_masked=cdl_already_masked,
                                     try_maximum_occurrence_approach=try_maximum_occurrence_approach_for_cdl,
                                     skip_processing=skip_cdl_cropland_processing)

    # irrigated cropland data processing
    # irrigated_crop_dict =

    # ssebop ET data processing
    ssebop_et_dict = \
        process_ssebop_data(input_ssebop_dir='../../Data_main/Raster_data/Ssebop_ETa/raw_data',
                            output_dir_ssebop_monthly='../../Data_main/Raster_data/Ssebop_ETa/WestUS_monthly',
                            output_dir_ssebop_yearly='../../Data_main/Raster_data/Ssebop_ETa/WestUS',
                            years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                   2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                            ref_raster=WestUS_raster,
                            skip_processing=skip_ssebop_processing)

    # prism precipitation data processing
    prism_precip_dict = \
        process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                  2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                           prism_bil_dir='../../Data_main/Raster_data/PRISM_PRECIP/bil_format',
                           prism_tif_dir='../../Data_main/Raster_data/PRISM_PRECIP/tif_format',
                           output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_PRECIP/WestUS_monthly',
                           output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_PRECIP/WestUS',
                           west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                           keyword='prism_precip', skip_processing=skip_prism_processing)

    # prism maximum temperature data processing
    prism_temp_dict = \
        process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                  2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                           prism_bil_dir='../../Data_main/Raster_data/PRISM_TEMP/bil_format',
                           prism_tif_dir='../../Data_main/Raster_data/PRISM_TEMP/tif_format',
                           output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_TEMP/WestUS_monthly',
                           output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_TEMP/WestUS',
                           west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                           keyword='prism_temp', skip_processing=skip_prism_processing)

    # return cdl_cropland_dict, ssebop_et_dict, irrigated_crop_dict, prism_precip_dict, prism_temp_dict, \
    #        terraclimate_dict, OpenET_processing_dict