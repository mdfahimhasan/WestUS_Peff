import os
import sys
import pickle
import numpy as np
import pandas as pd
from glob import glob
from osgeo import gdal
import geopandas as gpd

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.ml_ops import reindex_df
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


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
        # years and months to process over
        years = [2016, 2017, 2018, 2019, 2020]
        months = list(range(1, 13))

        # training zone bounding box gdf
        training_zone_gdf = gpd.read_file(training_zone_shp)

        # reference raster
        ref_arr, ref_file = read_raster_arr_object(refraster)
        total_bounds = ref_file.bounds

        # primary and secondary output directory creation
        shape_temp_output_dir = os.path.join(general_output_dir, 'shapes')
        cropET_interim_shape_dir = os.path.join(general_output_dir, 'cropET_interim_for_shapes')
        final_filtered_cropET_dir = os.path.join(general_output_dir, 'final_filtered_cropET_for_training')

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

        # collecting all rainfed cropET dataset
        rainfed_cropET_datasets = glob(os.path.join(rainfed_cropET_dir, '*.tif'))

        # going over each rainfed cropET data and applying the rasterized zone raster on it
        # this will create individual raster for each cropET raster for each bounding box
        for cropET in rainfed_cropET_datasets:
            cropET_arr, file = read_raster_arr_object(cropET)

            year = os.path.basename(cropET).split('_')[2]
            month = os.path.basename(cropET).split('_')[3].split('.')[0]

            rainfed_cropland_data = glob(os.path.join(rainfed_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            cdl_data = glob(os.path.join(usda_cdl_dir, f'*{year}*.tif'))[0]
            excess_et_filter_data = glob(os.path.join(excess_ET_filter_dir, f'*{year}*.tif'))[0]

            rainfed_cropland_arr = read_raster_arr_object(rainfed_cropland_data, get_file=False)
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)
            cdl_arr = read_raster_arr_object(cdl_data, get_file=False)
            excess_et_arr = read_raster_arr_object(excess_et_filter_data, get_file=False)

            # training zones types -
            # type 1: pasture lands with rainfed croplands or any rainfed cropland with no overlapping with irrigated croplands,
            #         excess_et_filter = 1 in both cases

            # collecting the train zone rasters and looping over them to apply each train zone raster on a cropET raster
            zone_rasters = glob(os.path.join(shape_temp_output_dir, '*.tif'))

            for zone_ras in zone_rasters:
                zone_id = os.path.basename(zone_ras).split('_')[2].split('.')[0]
                zone_arr = read_raster_arr_object(zone_ras, get_file=False)

                filtered_cropET_arr = np.where(((cdl_arr == 176) & (rainfed_cropland_arr == 1) & (excess_et_arr == 1)) |
                                               ((rainfed_cropland_arr == 1) & (irrigated_cropland_arr == -9999) &
                                                (excess_et_arr == 1)), cropET_arr, -9999)

                # applying the zone raster on the filtered cropET so that everything outside the box becomes nodata
                final_cropET_arr = filtered_cropET_arr * zone_arr

                # saving the cropET raster for each year-month for each train zone
                output_cropET = os.path.join(cropET_interim_shape_dir, f'cropET_zone_{zone_id}_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_cropET_arr, raster_file=file, transform=file.transform,
                                      output_path=output_cropET)

        # compiling final filtered training data
        for year in years:
            for month in months:
                print(f'compiling filtered effect. precip training data for year {year}, month {month}...')

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


def create_monthly_dataframes_for_eff_precip_prediction(years_list, month_range,
                                                        monthly_data_path_dict, yearly_data_path_dict,
                                                        static_data_path_dict, datasets_to_include, output_dir,
                                                        skip_processing=False):
    """
    Create monthly dataframes of predictors to generate monthly effective prediction.

    :param years_list: A list of years for which data to include in the dataframe.
    :param month_range: A tuple of start and end month for which data to filter. Set to None if there is no monthly dataset.
    :param monthly_data_path_dict: A dictionary with monthly variables' names as keys and their paths as values.
    :param yearly_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param static_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Set to None if there is no yearly dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this dataframe creation process.

    :return: None
    """
    if not skip_processing:
        makedirs([output_dir])

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

        for year in years_list:  # 1st loop controlling years
            for month in month_list:  # 2nd loop controlling months
                print(f'creating dataframe for prediction - year={year}, month={month}...')

                variable_dict = {}

                # reading monthly data and storing it in a dictionary
                for var in monthly_data_path_dict.keys():
                    if var in datasets_to_include:
                        monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]
                        data_arr = read_raster_arr_object(monthly_data, get_file=False).flatten()

                        data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                        variable_dict[var] = list(data_arr)
                        variable_dict['month'] = [int(month)] * len(data_arr)

                # reading monthly data and storing it in a dictionary
                if yearly_data_path_dict is not None:
                    for var in yearly_data_path_dict.keys():
                        if var in datasets_to_include:
                            yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]
                            data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                            data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                            variable_dict[var] = list(data_arr)

                # reading static data and storing it in a dictionary
                if static_data_path_dict is not None:
                    for var in static_data_path_dict.keys():
                        if var in datasets_to_include:
                            static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                            data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                            data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                            variable_dict[var] = list(data_arr)

                predictor_df = pd.DataFrame(variable_dict)
                predictor_df = predictor_df.dropna()

                if 'Rainfed_Frac' in predictor_df.columns:  # renaming 'Rainfed_Frac' as this attribute will not be
                    # available for Irrigated pixels
                    predictor_df = predictor_df.rename(columns={'Rainfed_Frac': 'Cropland_Frac'})

                # saving input predictor csv
                monthly_output_csv = os.path.join(output_dir, f'predictors_{year}_{month}.csv')
                predictor_df.to_csv(monthly_output_csv, index=False)

    else:
        pass


def create_nan_pos_dict_for_irrigated_cropET(irrigated_cropET_dir, output_dir, skip_processing=False):
    if not skip_processing:
        makedirs([output_dir])

        print('creating nan position dicts for irrigated cropET')

        irrigated_cropET_datasets = glob(os.path.join(irrigated_cropET_dir, '*.tif'))

        nan_pos_dict = {}  # an empty dict where nan positions will be saved as boolean data
        for data in irrigated_cropET_datasets:
            name = os.path.basename(data).split('.')[0]
            year = os.path.basename(data).split('_')[2]
            month = os.path.basename(data).split('_')[3].split('.')[0]

            arr = read_raster_arr_object(data, get_file=False).flatten()
            nan_pos_dict[name] = np.isnan(arr)

            nan_pos_dict_name = os.path.join(output_dir, f'nan_pos_{year}_{month}.pkl')
            pickle.dump(nan_pos_dict, open(nan_pos_dict_name, mode='wb+'))
    else:
        pass


def create_monthly_effective_precip_rasters(trained_model, input_csv_dir, exclude_columns,
                                            irrig_cropET_nan_pos_dir, ref_raster,
                                            prediction_name_keyword, output_dir, skip_processing=False):
    if not skip_processing:
        makedirs([output_dir])

        # ref raster shape
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_shape = ref_arr.shape

        # creating prediction raster for each month
        input_csvs = glob(os.path.join(input_csv_dir, '*.csv'))

        for csv in input_csvs:
            year = os.path.basename(csv).split('_')[1]
            month = os.path.basename(csv).split('_')[2].split('.')[0]
            print(f'Generating {prediction_name_keyword} prediction raster for year {year}, month {month}...')

            # loading input variable dataframe and filtering out columns
            df = pd.read_csv(csv)
            df = df.drop(columns=exclude_columns)
            df = reindex_df(df)

            # generating prediction with trained model
            pred_arr = trained_model.predict(df)
            pred_arr = np.array(pred_arr)

            # replacing values with -9999 where irrigated cropET is nan
            irrig_cropET_nan = glob(os.path.join(irrig_cropET_nan_pos_dir, f'*{year}_{month}.pkl*'))[0]
            nan_pos_dict = pickle.load(open(irrig_cropET_nan, mode='rb'))

            for var, nan_pos in nan_pos_dict.items():
                pred_arr[nan_pos_dict[var]] = ref_file.nodata

            # reshaping the prediction raster for Western US and saving
            pred_arr = pred_arr.reshape(ref_shape)
            # print(prediction_arr.shape)
            output_prediction_raster = os.path.join(output_dir, f'{prediction_name_keyword}_{year}_{month}.tif')
            write_array_to_raster(raster_arr=pred_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_prediction_raster)

    else:
        pass


def sum_monthly_effective_precip_rasters(years_list, monthly_effective_precip_dir,
                                         irrigated_cropET_dir,
                                         grow_season_effective_precip_output_dir,
                                         skip_processing=False):
    """
    Sum monthly effective precip predictions by the ML model to grow season effective precip.

    :param years_list: A list of years to process the data.
    :param monthly_effective_precip_dir: Directory path of model-generated effective precipitation rasters.
    :param irrigated_cropET_dir: Directory path of growing season irrigated cropET. Will be used to set 0 and no data
                                 values to effective precipitation dataset.
    :param grow_season_effective_precip_output_dir: Directory path of summed growing season effective
    precipitation rasters.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([grow_season_effective_precip_output_dir])

        for year in years_list:
            print(f'Summing effective precipitation monthly raster for grow season {year}...')

            input_rasters = glob(os.path.join(monthly_effective_precip_dir, f'*{year}*.tif'))

            sum_arr, file = None, None
            for raster in input_rasters:
                if raster == input_rasters[0]:
                    arr, file = read_raster_arr_object(raster)
                    sum_arr = arr
                else:
                    arr = read_raster_arr_object(raster, get_file=False)
                    sum_arr = np.nansum(np.dstack((sum_arr, arr)), axis=2)

            # # setting values over non-irrigated croplands to 0
            irrig_cropET = glob(os.path.join(irrigated_cropET_dir, f'*{year}*.tif'))[0]
            irrig_cropET_arr = read_raster_arr_object(irrig_cropET, get_file=False)

            sum_arr = np.where(np.isnan(irrig_cropET_arr), -9999, sum_arr)

            summed_raster = os.path.join(grow_season_effective_precip_output_dir, f'effective_precip_{year}.tif')
            write_array_to_raster(raster_arr=sum_arr, raster_file=file, transform=file.transform,
                                  output_path=summed_raster)



