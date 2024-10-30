import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.ml_ops import reindex_df
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, create_multiband_raster, sum_rasters

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def create_monthly_dataframes_for_eff_precip_prediction(years_list, month_range,
                                                        monthly_data_path_dict, yearly_data_path_dict,
                                                        static_data_path_dict, datasets_to_include, output_dir,
                                                        skip_processing=False):
    """
    Create monthly dataframes of predictors to generate monthly effective prediction.

    :param years_list: A list of years_list for which data to include in the dataframe.
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

        for year in years_list:  # 1st loop controlling years_list
            for month in month_list:  # 2nd loop controlling months

                if year == 1999 and month in range(1, 10):  # skipping dataframe creation for 1999 January-September
                    continue

                else:
                    print(f'creating dataframe for prediction - year={year}, month={month}...')

                    variable_dict = {}

                    # reading monthly data and storing it in a dictionary
                    for var in monthly_data_path_dict.keys():
                        if var in datasets_to_include:

                            if var == 'GRIDMET_Precip':  # for including monthly and lagged monthly GRIDMET_precip in the dataframe
                                current_precip_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]

                                current_month_date = datetime(year, month, 1)

                                # Collect previous month's precip data
                                prev_month_date = current_month_date - timedelta(30)
                                prev_2_month_date = current_month_date - timedelta(60)

                                prev_month_precip_data = glob(os.path.join(monthly_data_path_dict[var],
                                                                           f'*{prev_month_date.year}_{prev_month_date.month}.tif*'))[0]
                                prev_2_month_precip_data = glob(os.path.join(monthly_data_path_dict[var],
                                                                             f'*{prev_2_month_date.year}_{prev_2_month_date.month}.tif*'))[0]

                                # reading datasets
                                current_precip_arr = read_raster_arr_object(current_precip_data, get_file=False).flatten()

                                prev_month_precip_arr = read_raster_arr_object(prev_month_precip_data, get_file=False).flatten()
                                prev_2_month_precip_arr = read_raster_arr_object(prev_2_month_precip_data, get_file=False).flatten()

                                current_precip_arr[np.isnan(current_precip_arr)] = 0  # setting nan-position values with 0
                                prev_month_precip_arr[np.isnan(prev_month_precip_arr)] = 0  # setting nan-position values with 0
                                prev_2_month_precip_arr [np.isnan(prev_2_month_precip_arr)] = 0  # setting nan-position values with 0

                                variable_dict[var] = list(current_precip_arr)
                                variable_dict['month'] = [int(month)] * len(current_precip_arr)
                                variable_dict['GRIDMET_Precip_1_lag'] = list(prev_month_precip_arr)
                                variable_dict['GRIDMET_Precip_2_lag'] = list(prev_2_month_precip_arr)

                            else:
                                monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]
                                data_arr = read_raster_arr_object(monthly_data, get_file=False).flatten()

                                data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                                variable_dict[var] = list(data_arr)
                                variable_dict['month'] = [int(month)] * len(data_arr)

                # reading yearly data and storing it in a dictionary
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

                # saving input predictor csv
                monthly_output_csv = os.path.join(output_dir, f'predictors_{year}_{month}.csv')
                predictor_df.to_csv(monthly_output_csv, index=False)

    else:
        pass


def create_nan_pos_dict_for_monthly_irrigated_cropET(irrigated_cropET_dir, output_dir, skip_processing=False):
    """
    Stores nan position pkl files for monthly irrigated cropET datasets.

    :param irrigated_cropET_dir: Filepath of input monthly irrigated cropET directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to true to skip this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        print('creating nan position dicts for irrigated cropET...')

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
                                            irrig_cropET_nan_pos_dir,
                                            prediction_name_keyword, output_dir,
                                            ref_raster=WestUS_raster,skip_processing=False):
    """
    Create monthly effective precipitation prediction raster.

    :param trained_model: Trained ML model object.
    :param input_csv_dir: Filepath of input directory consisting of monthly predictor csvs for the model.
    :param exclude_columns: List of predictors to exclude from model prediction.
    :param irrig_cropET_nan_pos_dir: Filepath of input directory consisting of monthly nan position (irrigated cropET)
                                     pkl files.
    :param prediction_name_keyword: A str that will be added before prediction file name.
    :param output_dir: Filepath of output directory to store predicted rasters.
    :param ref_raster: Filepath of ref raster. Default set to WestUS reference raster.
    :param skip_processing: Set to true to skip this processing step.

    :return: None.
    """

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

            nan_key = f'Irrigated_cropET_{year}_{month}'
            pred_arr[nan_pos_dict[nan_key]] = ref_file.nodata

            # reshaping the prediction raster for Western US and saving
            pred_arr = pred_arr.reshape(ref_shape)

            output_prediction_raster = os.path.join(output_dir, f'{prediction_name_keyword}_{year}_{month}.tif')
            write_array_to_raster(raster_arr=pred_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_prediction_raster)

    else:
        pass


def create_annual_dataframes_for_peff_frac_prediction(years_list, yearly_data_path_dict,
                                                      static_data_path_dict, datasets_to_include, output_dir,
                                                      skip_processing=False):
    """
    Create monthly dataframes of predictors to generate monthly effective prediction.

    :param years_list: A list of years_list for which data to include in the dataframe.
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

        for year in years_list:  # 1st loop controlling years_list
                print(f'creating dataframe for prediction - year={year}...')

                variable_dict = {}

                # reading yearly data and storing it in a dictionary
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

                # saving input predictor csv
                monthly_output_csv = os.path.join(output_dir, f'predictors_{year}.csv')
                predictor_df.to_csv(monthly_output_csv, index=False)

    else:
        pass


def create_nan_pos_dict_for_annual_irrigated_cropET(irrigated_cropET_dir, output_dir, skip_processing=False):
    """
    Stores nan position pkl files for annual/water year irrigated cropET datasets.

    :param irrigated_cropET_dir: Filepath of input annual/water year irrigated cropET directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to true to skip this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        print('creating nan position dicts for irrigated cropET...')

        irrigated_cropET_datasets = glob(os.path.join(irrigated_cropET_dir, '*.tif'))

        nan_pos_dict = {}  # an empty dict where nan positions will be saved as boolean data
        for data in irrigated_cropET_datasets:
            name = os.path.basename(data).split('.')[0]
            year = os.path.basename(data).split('_')[2].split('.')[0]

            arr = read_raster_arr_object(data, get_file=False).flatten()
            nan_pos_dict[name] = np.isnan(arr)

            nan_pos_dict_name = os.path.join(output_dir, f'nan_pos_{year}.pkl')
            pickle.dump(nan_pos_dict, open(nan_pos_dict_name, mode='wb+'))
    else:
        pass


def create_annual_peff_fraction_rasters(trained_model, input_csv_dir, exclude_columns,
                                        irrig_cropET_nan_pos_dir,
                                        lake_raster, ref_raster,
                                        prediction_name_keyword, output_dir,
                                        skip_processing=False):
    """
    Create annual/water year effective precipitation fraction prediction raster.

    :param trained_model: Trained ML model object.
    :param input_csv_dir: Filepath of input directory consisting of annual/water year predictor csvs for the model.
    :param exclude_columns: List of predictors to exclude from model prediction.
    :param irrig_cropET_nan_pos_dir: Filepath of input directory consisting of annual.water year nan position
                                     (irrigated cropET) pkl files.
    :param prediction_name_keyword: A str that will be added before prediction file name.
    :param output_dir: Filepath of output directory to store predicted rasters.
    :param lake_raster: Filepath of lake raster.
    :param ref_raster: Filepath of ref raster. Default set to WestUS reference raster.
    :param skip_processing: Set to true to skip this processing step.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # ref raster shape
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_shape = ref_arr.shape

        # loading lake raster data
        lake_arr = read_raster_arr_object(lake_raster, get_file=False)

        # creating prediction raster for each month
        input_csvs = glob(os.path.join(input_csv_dir, '*.csv'))

        for csv in input_csvs:
            year = os.path.basename(csv).split('_')[1].split('.')[0]

            print(f'Generating {prediction_name_keyword} prediction raster for year {year}...')

            # loading input variable dataframe and filtering out columns
            df = pd.read_csv(csv)
            df = df.drop(columns=exclude_columns)
            df = reindex_df(df)

            # generating prediction with trained model
            pred_arr = trained_model.predict(df)
            pred_arr = np.array(pred_arr)

            # replacing >1 fraction values with 1. From our observation, the number of values replaced with this
            # filtering approach isn't much
            pred_arr = np.where(pred_arr > 1, 1, pred_arr)

            # applying water body masking with lake raster
            pred_arr = np.where(lake_arr == 1, -9999, pred_arr)

            # replacing values with -9999 where irrigated cropET is nan
            irrig_cropET_nan = glob(os.path.join(irrig_cropET_nan_pos_dir, f'*{year}.pkl*'))[0]
            nan_pos_dict = pickle.load(open(irrig_cropET_nan, mode='rb'))

            nan_key = f'Irrigated_cropET_{year}'
            pred_arr[nan_pos_dict[nan_key]] = ref_file.nodata

            # reshaping the prediction raster for Western US and saving
            pred_arr = pred_arr.reshape(ref_shape)

            output_prediction_raster = os.path.join(output_dir, f'{prediction_name_keyword}_{year}.tif')
            write_array_to_raster(raster_arr=pred_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_prediction_raster)

    else:
        pass


def collect_Peff_predictions_in_dataframe(input_peff_dir, output_csv, skip_processing=False):
    """
    Gathering monthly effective precipitation or annual effective precipitation fraction predictions into csv.

    :param input_peff_dir: Filepath of monthly effective precipitation or annual effective precipitation fraction
                           prediction.
    :param output_csv: Filepath of output csv.
    :param skip_processing: Set to True if want to skip this step.

    :return: None.
    """
    if not skip_processing:
        peff_rasters = glob(os.path.join(input_peff_dir, '*.tif'))

        # an empty list to store the data
        peff = []

        for each in peff_rasters:
            arr = read_raster_arr_object(each, get_file=False).flatten()
            peff.extend(arr.tolist())

        # converting to a dataframe
        peff_df = pd.DataFrame({'peff': peff})
        peff_df = peff_df.dropna()

        peff_df.to_csv(output_csv, index=False)
    else:
        pass


def sum_peff_water_year(years_list, monthly_peff_dir, output_peff_dir, skip_processing=False):
    """
    Sum monthly effective precipitation estimates for water year.

    :param years_list: List of years to process data for.
    :param monthly_peff_dir: Directory path of monthly effective precipitation.
    :param output_peff_dir: Directory path of summed water year effective precipitation.
    :param skip_processing: Set to False if want to skip this step.

    :return: None.
    """
    if not skip_processing:
        print('Summing water year effective precipitation...')

        makedirs([output_peff_dir])

        for yr in years_list:
            # # summing peff for water year (previous year's October to current year's september)
            peff_prev_years = glob(os.path.join(monthly_peff_dir, f'*{yr - 1}_1[0-2].*tif'))
            pff_data_current_years = glob(os.path.join(monthly_peff_dir, f'*{yr}_[1-9].*tif'))
            peff_water_yr_list = peff_prev_years + pff_data_current_years

            sum_rasters(raster_list=peff_water_yr_list, raster_dir=None,
                        output_raster=os.path.join(output_peff_dir, f'effective_precip_{yr}.tif'),
                        ref_raster=peff_water_yr_list[0])
    else:
        pass


def estimate_peff_precip_water_year_fraction(years_list, peff_dir_water_yr, precip_dir_water_yr, output_dir,
                                             skip_processing=False):
    """
    Estimated fraction of effective precipitation to precipitation for water years_list.

    :param years_list: List of years_list to process data.
    :param peff_dir_water_yr: Directory path of water year effective precipitation.
    :param precip_dir_water_yr: Directory path of water year precipitation.
    :param output_dir: File path of output directory.
    :param skip_processing:  Set to False if want to skip this step.

    :return: None.
    """
    if not skip_processing:
        print('Estimating fraction of water year effective precipitation to precipitation...')

        for yr in years_list:
            # collecting and reading datasets
            peff_data = glob(os.path.join(peff_dir_water_yr, f'*{yr}*.tif'))[0]
            precip_data = glob(os.path.join(precip_dir_water_yr, f'*{yr}*.tif'))[0]

            peff_arr, file = read_raster_arr_object(peff_data)
            precip_arr = read_raster_arr_object(precip_data, get_file=False)

            # estimating peff/precip fraction over valid pixels
            frac_arr = np.where(peff_arr != -9999, peff_arr/precip_arr, -9999)

            # saving raster
            makedirs([output_dir])

            output_raster = os.path.join(output_dir, f'peff_frac_{yr}.tif')
            write_array_to_raster(frac_arr, file, file.transform, output_raster)
    else:
        pass


def process_monthly_peff_rasters_to_multiband_forGEE(years, peff_monthly_dir, output_dir, nodata=no_data_value):
    """
    Compiles monthly effective precipitation estimates into multi-band rasters for each year.

    :param years: List of years_list.
    :param peff_monthly_dir: Filepath of monthly effective precipitation rasters.
    :param output_dir: Filepath of output directory to save the multi-band raster.
    :param nodata: Default set to -9999.

    :return: None,
    """
    makedirs([output_dir])
    for year in years:
        peff_data_list = []

        # collecting the monthly Peff estimates serially for a year
        for month in list(range(4, 11)):  # list of months from 4-10
            monthly_peff = glob(os.path.join(peff_monthly_dir, f'*{year}_{month}*.tif'))
            peff_data_list.append(monthly_peff[0])

        # creating the multi-band image for monthly datasets within an year
        output_raster = os.path.join(output_dir, f'effective_precip_{year}_monthly.tif')

        create_multiband_raster(input_files_list=peff_data_list,
                                output_file=output_raster, nodata=nodata)
        print(f'created effective precipitation monthly multi-band raster for {year}...')


def estimate_water_yr_peff_using_peff_frac(years_list, water_year_precip_dir, water_year_peff_frac_dir, output_dir,
                                           skip_processing=False):
    """
    Estimate water year effective precipitation (peff) using water year peff fraction.

    :param years_list: List of years_list to process data for.
    :param water_year_precip_dir: Filepath of water year precipitation directory.
    :param water_year_peff_frac_dir: Filepath of water year effective precipitation fraction (from water year model)
                                     directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this process. Default set to False.

    :return: None.
    """

    if not skip_processing:
        makedirs([output_dir])

        for yr in years_list:
            print(f'Estimating water year Peff using water year Peff fraction (from water year model) for year {yr}...')

            # laoding and reading data
            precip = glob(os.path.join(water_year_precip_dir, f'*{yr}*.tif'))[0]
            peff_frac = glob(os.path.join(water_year_peff_frac_dir, f'*{yr}*.tif'))[0]

            precip_arr = read_raster_arr_object(precip, get_file=False)
            peff_frac_arr, raster_file = read_raster_arr_object(peff_frac)

            # bounding peff_fraction value of 1 where the fraction exceeds 1
            peff_frac_arr[peff_frac_arr > 1] = 1

            # estimating peff water year total
            new_arr = np.where(~np.isnan(peff_frac_arr) & ~np.isnan(precip_arr), precip_arr * peff_frac_arr, -9999)

            output_raster = os.path.join(output_dir, f'peff_water_year_{yr}.tif')
            write_array_to_raster(new_arr, raster_file, raster_file.transform, output_raster)
    else:
        pass


def scale_monthy_peff_with_wateryr_peff_model(years_list, unscaled_peff_monthly_dir, unscaled_peff_water_yr_dir,
                                              scaled_peff_water_yr_dir, output_dir,
                                              skip_processing=False):
    """
    Scale effective precipitation (peff) monthly data using the water year peff fraction model.
    The model ensures imposing water year precipitation > water year peff

    :param years_list: List of years_list to process data for.
    :param unscaled_peff_monthly_dir: Filepath of original monthly peff (from monthly model) estimates' directory.
    :param unscaled_peff_water_yr_dir: Filepath of original water year peff, summed from monthly model, estimates'
                                       directory.
    :param scaled_peff_water_yr_dir: Filepath of updated water year peff, estimated from water year model, estimates'
                                     directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this process. Default set to False.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        months = range(1, 12 + 1)

        for yr in years_list:
            for mn in months:

                if (yr == 1999) and (mn <= 9):
                    continue

                elif (yr == 2020) and (mn >= 10):
                    continue

                else:
                    print(f'Scaling monthly Peff with water year Peff fraction model for year {yr}, month {mn}...')

                    # selecting the water year of total peff data based on month
                    if mn in range(10, 12 + 1):
                        peff_unbound_wy = glob(os.path.join(unscaled_peff_water_yr_dir, f'*{yr + 1}*.tif'))[0]
                        peff_unbound_wy_arr = read_raster_arr_object(peff_unbound_wy, get_file=False)

                        peff_bound_wy = glob(os.path.join(scaled_peff_water_yr_dir, f'*{yr + 1}*.tif'))[0]
                        peff_bound_wy_arr = read_raster_arr_object(peff_bound_wy, get_file=False)

                    elif mn in range(1, 9 + 1):
                        peff_unbound_wy = glob(os.path.join(unscaled_peff_water_yr_dir, f'*{yr}*.tif'))[0]
                        peff_unbound_wy_arr = read_raster_arr_object(peff_unbound_wy, get_file=False)

                        peff_bound_wy = glob(os.path.join(scaled_peff_water_yr_dir, f'*{yr}*.tif'))[0]
                        peff_bound_wy_arr = read_raster_arr_object(peff_bound_wy, get_file=False)

                    # selecting the monthly peff data
                    unscaled_peff_monthly = glob(os.path.join(unscaled_peff_monthly_dir, f'*{yr}_{mn}*.tif'))[0]
                    unscaled_peff_monthly_arr, raster_file = read_raster_arr_object(unscaled_peff_monthly)

                    # scaling monthly peff with bounded peff total
                    # the bounded peff total comes from water year model, where we impose water year peff < water year precip
                    scaled_peff_monthly_arr = np.where(~np.isnan(unscaled_peff_monthly_arr),
                                                       unscaled_peff_monthly_arr * peff_bound_wy_arr / peff_unbound_wy_arr,
                                                       -9999)
                    scaled_peff_monthly_arr[unscaled_peff_monthly_arr == -9999] = -9999

                    output_raster = os.path.join(output_dir, f'effective_precip_{yr}_{mn}.tif')
                    write_array_to_raster(scaled_peff_monthly_arr, raster_file, raster_file.transform, output_raster)