import os
import sys
import csv
import joblib
import timeit
import numpy as np
import pandas as pd
from glob import glob
import dask.dataframe as ddf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from timeit import default_timer as timer

import lightgbm as lgb
from lightgbm import LGBMRegressor

from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

import skexplain
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay as PDisp

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_rmse, calculate_r2
from Codes.utils.raster_ops import read_raster_arr_object

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'


def reindex_df(df):
    """
    Reindex dataframe based on column names.

    :param df: Predictor dataframe.

    :return: Reindexed dataframe.
    """
    sorted_columns = sorted(df.columns)
    df = df.reindex(sorted_columns, axis=1)

    return df


def apply_OneHotEncoding(input_df):
    one_hot = OneHotEncoder()
    input_df_enc = one_hot.fit_transform(input_df)
    return input_df_enc


def create_train_test_monthly_dataframe(years_list, monthly_data_path_dict, yearly_data_path_dict,
                                        static_data_path_dict, datasets_to_include, output_parquet,
                                        skip_processing=False, n_partitions=20):
    """
    Compile monthly/yearly/static datasets into a dataframe. This function-generated dataframe will be used as
    train-test data for ML model at monthly scale.

    *** if there is no yearly dataset, set yearly_data_path_dict to None.
    *** if there is no static data, set static_data_path_dict to None.

    :param years_list: A list of years_list for which data to include in the dataframe.
    :param monthly_data_path_dict: A dictionary with monthly variables' names as keys and their paths as values.
                                   This can't be None.
    :param yearly_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Set to None if there is no yearly dataset.
    :param static_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_parquet: Output filepath of the parquet file to save. Using parquet as it requires lesser memory.
                            Can also save smaller dataframe as csv file if name has '.csv' extension.
    :param skip_processing: Set to True to skip this dataframe creation process.
    :param n_partitions: Number of partitions to save the parquet file in using dask dataframe.

    :return: The filepath of the output parquet file.
    """
    if not skip_processing:
        print('creating train-test dataframe for monthly model...')

        output_dir = os.path.dirname(output_parquet)
        makedirs([output_dir])

        variable_dict = {}
        yearly_month_count_dict = {}

        # monthly data compilation
        for var in monthly_data_path_dict.keys():
            if var in datasets_to_include:
                print(f'processing data for {var}...')

                for year in years_list:
                    # creating list of month to be included for each year
                    if year == 2008:
                        month_list = range(10, 13)
                    elif year == 2020:
                        month_list = range(1, 10)
                    else:
                        month_list = range(1, 13)

                    if var == 'GRIDMET_Precip':  # for including monthly and lagged monthly GRIDMET_precip in the dataframe
                        for month_count, month in enumerate(month_list):
                            current_precip_data = \
                            glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]

                            current_month_date = datetime(year, month, 1)

                            # Collect previous month's precip data
                            prev_month_date = current_month_date - timedelta(30)
                            prev_2_month_date = current_month_date - timedelta(60)

                            prev_month_precip_data = glob(os.path.join(monthly_data_path_dict[var],
                                                                       f'*{prev_month_date.year}_{prev_month_date.month}.tif*'))[
                                0]
                            prev_2_month_precip_data = glob(os.path.join(monthly_data_path_dict[var],
                                                                         f'*{prev_2_month_date.year}_{prev_2_month_date.month}.tif*'))[
                                0]

                            # reading datasets
                            current_precip_arr = read_raster_arr_object(current_precip_data, get_file=False).flatten()
                            len_arr = len(list(current_precip_arr))
                            year_data = [int(year)] * len_arr
                            month_data = [int(month)] * len_arr

                            prev_month_precip_arr = read_raster_arr_object(prev_month_precip_data,
                                                                           get_file=False).flatten()
                            prev_2_month_precip_arr = read_raster_arr_object(prev_2_month_precip_data,
                                                                             get_file=False).flatten()

                            if (month_count == 0) & (
                                    var not in variable_dict.keys()):  # initiating the key and adding first series of data
                                variable_dict[var] = list(current_precip_arr)
                                variable_dict['year'] = year_data
                                variable_dict['month'] = month_data

                                variable_dict['GRIDMET_Precip_1_lag'] = list(prev_month_precip_arr)
                                variable_dict['GRIDMET_Precip_2_lag'] = list(prev_2_month_precip_arr)

                            else:  # if key already found, extending the data list
                                variable_dict[var].extend(list(current_precip_arr))
                                variable_dict['year'].extend(year_data)
                                variable_dict['month'].extend(month_data)

                                variable_dict['GRIDMET_Precip_1_lag'].extend(list(prev_month_precip_arr))
                                variable_dict['GRIDMET_Precip_2_lag'].extend(list(prev_2_month_precip_arr))

                            yearly_month_count_dict[year] = month_count + 1

                    else:
                        for month_count, month in enumerate(month_list):
                            monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]

                            data_arr = read_raster_arr_object(monthly_data, get_file=False).flatten()
                            len_arr = len(list(data_arr))
                            year_data = [int(year)] * len_arr
                            month_data = [int(month)] * len_arr

                            if (month_count == 0) & (
                                    var not in variable_dict.keys()):  # initiating the key and adding first series of data
                                variable_dict[var] = list(data_arr)
                                variable_dict['year'] = year_data
                                variable_dict['month'] = month_data
                            else:  # if key already found, extending the data list
                                variable_dict[var].extend(list(data_arr))
                                variable_dict['year'].extend(year_data)
                                variable_dict['month'].extend(month_data)

                            yearly_month_count_dict[year] = month_count + 1

        # annual data compilation
        if yearly_data_path_dict is not None:
            for var in yearly_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}..')

                    for year_count, year in enumerate(years_list):
                        yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]

                        data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                        if (year_count == 0) & (var not in variable_dict.keys()):
                            variable_dict[var] = list(data_arr) * yearly_month_count_dict[year]

                        else:
                            variable_dict[var].extend(list(data_arr) * yearly_month_count_dict[year])

        # static data compilation
        # counting total number of months in all years_list
        # need only for monthly train-test dataframe creation
        total_month_count = 0
        for i in yearly_month_count_dict.values():
            total_month_count += i

        if static_data_path_dict is not None:
            for var in static_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}..')

                    static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]

                    data_arr = read_raster_arr_object(static_data, get_file=False).flatten()
                    data_duplicated_for_total_months = list(data_arr) * total_month_count

                    variable_dict[var] = data_duplicated_for_total_months

        train_test_ddf = ddf.from_dict(variable_dict, npartitions=n_partitions)
        train_test_ddf = train_test_ddf.dropna()

        if '.parquet' in output_parquet:
            train_test_ddf.to_parquet(output_parquet, write_index=False)
        elif '.csv' in output_parquet:
            train_test_df = train_test_ddf.compute()
            train_test_df.to_csv(output_parquet, index=False)

        return output_parquet

    else:
        return output_parquet


def create_train_test_annual_dataframe(years_list, yearly_data_path_dict,
                                       static_data_path_dict, datasets_to_include, output_parquet,
                                       skip_processing=False, n_partitions=20):
    """
    Compile yearly/static datasets into a dataframe. This function-generated dataframe will be used as
    train-test data for ML model at annual scale.

    *** if there is no static data, set static_data_path_dict to None.

    :param years_list: A list of years_list for which data to include in the dataframe.
    :param yearly_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Can't be None.
    :param static_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_parquet: Output filepath of the parquet file to save. Using parquet as it requires lesser memory.
                            Can also save smaller dataframe as csv file if name has '.csv' extension.
    :param skip_processing: Set to True to skip this dataframe creation process.
    :param n_partitions: Number of partitions to save the parquet file in using dask dataframe.
    :param filter_for_slope: Se to True to filter for slope. Default set to False to skip the filtering.

    :return: The filepath of the output parquet file.
    """
    if not skip_processing:
        print('creating train-test dataframe for annual model...')

        output_dir = os.path.dirname(output_parquet)
        makedirs([output_dir])

        variable_dict = {}

        # annual data compilation
        for var in yearly_data_path_dict.keys():
            if var in datasets_to_include:
                print(f'processing data for {var}..')

                for year_count, year in enumerate(years_list):
                    yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]

                    data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                    if (year_count == 0) & (var not in variable_dict.keys()):
                        variable_dict[var] = list(data_arr)

                    else:
                        variable_dict[var].extend(list(data_arr))

        # static data compilation
        if static_data_path_dict is not None:
            for var in static_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}..')

                    static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                    data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                    data_duplicated_for_total_years = list(data_arr) * len(years_list)
                    variable_dict[var] = data_duplicated_for_total_years

        train_test_ddf = ddf.from_dict(variable_dict, npartitions=n_partitions)
        train_test_ddf = train_test_ddf.dropna()

        if '.parquet' in output_parquet:
            train_test_ddf.to_parquet(output_parquet, write_index=False)
        elif '.csv' in output_parquet:
            train_test_df = train_test_ddf.compute()
            train_test_df.to_csv(output_parquet, index=False)

        return output_parquet

    else:
        return output_parquet


def split_train_val_test_set(input_csv, pred_attr, exclude_columns, output_dir, model_version,
                             month_range=None, test_perc=0.3, validation_perc=0,
                             random_state=0, verbose=True, remove_outlier=False,
                             outlier_upper_val=None, skip_processing=False):
    """
    Split dataset into train, validation, and test data based on a train/test/validation ratio.


    :param input_csv : Input csv file (with filepath) containing all the predictors.
    :param pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    :param exclude_columns : Tuple of columns that will not be included in training the fitted_model.
    :param output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    :param model_version: Model version name. Can be 'v1' or 'v2'.
    :param month_range: A tuple of start and end month for which data to filter. Default set to None.
    :param test_perc : The percentage of test dataset. Defaults to 0.3.
    :param validation_perc : The percentage of validation dataset. Defaults to 0.
    :param random_state : Seed value. Defaults to 0.
    :param verbose : Set to True if want to print which columns are being dropped and which will be included
                     in the model.
    :param remove_outlier: Set to True if we want to consider outlier removal while making the train-test split.
    :param outlier_upper_val: The upper outlier detection range from IQR or MAD.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    global x_val, y_val

    if not skip_processing:
        print('Splitting train-test dataframe into train and test dataset...')

        input_df = pd.read_parquet(input_csv)

        if month_range is not None:  # filter for specific month ranges
            month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
            input_df = input_df[input_df['month'].isin(month_list)]

        if remove_outlier:  # removing outliers. detected by EDA
            input_df = input_df[input_df[pred_attr] <= outlier_upper_val]

        # dropping columns that has been specified to not include
        drop_columns = exclude_columns + [
            pred_attr]  # dropping unwanted columns/columns that will not be used in model training
        x = input_df.drop(columns=drop_columns)
        y = input_df[pred_attr]  # response attribute

        # Reindexing for ensuring that columns go into the model in same serial every time
        x = reindex_df(x)

        if verbose:
            print('Dropping Columns-', exclude_columns, '\n')
            print('Predictors:', x.columns)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_perc, random_state=random_state,
                                                            shuffle=True)
        if validation_perc > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_perc,
                                                              random_state=random_state, shuffle=True)

        # creating dataframe and saving train/test/validation datasets as csv
        makedirs([output_dir])

        x_train_df = pd.DataFrame(x_train)
        x_train_df.to_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'), index=False)

        y_train_df = pd.DataFrame(y_train)
        y_train_df.to_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'), index=False)

        x_test_df = pd.DataFrame(x_test)
        x_test_df.to_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'), index=False)

        y_test_df = pd.DataFrame(y_test)
        y_test_df.to_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'), index=False)

        if validation_perc > 0:
            x_val_df = pd.DataFrame(x_val)
            x_val_df.to_csv(os.path.join(output_dir, f'x_val_{model_version}.csv'), index=False)

            y_val_df = pd.DataFrame(y_val)
            y_val_df.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)

        if validation_perc == 0:
            return x_train, x_test, y_train, y_test
        else:
            return x_train, x_val, x_test, y_train, y_val, y_test

    else:
        if validation_perc == 0:
            x_train = pd.read_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'))
            x_test = pd.read_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'))
            y_train = pd.read_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'))
            y_test = pd.read_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'))

            return x_train, x_test, y_train, y_test

        else:
            x_train = pd.read_csv(os.path.join(output_dir, f'x_train_{model_version}.csv'))
            x_test = pd.read_csv(os.path.join(output_dir, f'x_test_{model_version}.csv'))
            x_val = pd.read_csv(os.path.join(output_dir, f'x_val_{model_version}.csv'))
            y_train = pd.read_csv(os.path.join(output_dir, f'y_train_{model_version}.csv'))
            y_test = pd.read_csv(os.path.join(output_dir, f'y_test_{model_version}.csv'))
            y_val = pd.read_csv(os.path.join(output_dir, f'y_val_{model_version}.csv'))

            return x_train, x_val, x_test, y_train, y_val, y_test


def split_train_val_test_set_by_year(input_csv, pred_attr, exclude_columns,
                                     years_in_train, year_in_test, output_dir,
                                     verbose=True, skip_processing=False):
    """
    Split dataset into train, validation, and test data based on a train/test/validation ratio.

    :param input_csv : Input csv file (with filepath) containing all the predictors.
    :param pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    :param exclude_columns : Tuple of columns that will not be included in training the fitted_model.
    :param years_in_train: List of years_list to keep as train dataset. Input multiple years_list.
    :param year_in_test: List of year to keep as test dataset. Input single year.
    :param output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    :param verbose : Set to True if want to print which columns are being dropped and which will be included
                     in the model.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    if not skip_processing:
        print(f'Making train-test split with...', '\n',
              f'years_list {years_in_train} in train set', '\n',
              f'year {year_in_test} in test set')

        input_df = pd.read_parquet(input_csv)
        drop_columns = exclude_columns + [
            pred_attr]  # dropping unwanted columns/columns that will not be used in model training

        # making train-test split based on provided years_list
        train_df = input_df[input_df['year'].isin(years_in_train)]
        test_df = input_df[input_df['year'].isin(year_in_test)]

        x_train_df = train_df.drop(columns=drop_columns)
        y_train_df = train_df[pred_attr]

        x_test_df = test_df.drop(columns=drop_columns)
        y_test_df = test_df[pred_attr]

        # Reindexing for ensuring that columns go into the model in same serial every time
        x_train_df = reindex_df(x_train_df)
        x_test_df = reindex_df(x_test_df)

        if verbose:
            print('Dropping Columns-', exclude_columns, '\n')
            print('Predictors:', x_train_df.columns)

        # creating dataframe and saving train/test/validation datasets as csv
        makedirs([output_dir])

        x_train_df.to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
        y_train_df.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        x_test_df.to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)
        y_test_df.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    else:
        x_train_df = pd.read_csv(os.path.join(output_dir, 'x_train.csv'))
        x_test_df = pd.read_csv(os.path.join(output_dir, 'x_test.csv'))
        y_train_df = pd.read_csv(os.path.join(output_dir, 'y_train.csv'))
        y_test_df = pd.read_csv(os.path.join(output_dir, 'y_test.csv'))

    return x_train_df, x_test_df, y_train_df, y_test_df


def objective_func_bayes(params, train_set, iteration_csv, n_fold):
    """
    Objective function for Bayesian optimization using Hyperopt and LightGBM.

    :param params: Hyperparameter space to use while optimizing.
    :param train_set: A LGBM dataset. Constructed within the bayes_hyperparam_opt() func using x_train and y_train.
    :param iteration_csv : Filepath of a csv where hyperparameter iteration step will be stored.
    :param n_fold : KFold cross validation number. Usually 5 or 10.

    :return : A dictionary after each iteration holding rmse, params, run_time, etc.
    """
    global ITERATION
    ITERATION += 1

    start = timer()

    # converting the train_set (dataframe) to LightGBM Dataset
    train_set = lgb.Dataset(train_set.iloc[:, :-1], label=train_set.iloc[:, -1])

    # retrieve the boosting type and subsample (if not present set subsample to 1)
    subsample = params['boosting_type'].get('subsample', 1)
    params['subsample'] = subsample
    params['boosting_type'] = params['boosting_type']['boosting_type']

    # inserting a new parameter in the dictionary to handle 'goss'
    # the new version of LIGHTGBM handles 'goss' as 'boosting_type' = 'gdbt' & 'data_sample_strategy' = 'goss'
    if params['boosting_type'] == 'goss':
        params['boosting_type'] = 'gbdt'
        params['data_sample_strategy'] = 'goss'

    # ensure integer type for integer hyperparameters
    for parameter_name in ['n_estimators', 'num_leaves', 'min_child_samples', 'max_depth']:
        params[parameter_name] = int(params[parameter_name])

    # callbacks
    callbacks = [
        # lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)
    ]

    # perform n_fold cross validation
    # ** not using num_boost_round and early stopping as we are providing n_estimators in the param_space **
    cv_results = lgb.cv(params, train_set,
                        # num_boost_round=10000,
                        nfold=n_fold,
                        stratified=False, metrics='rmse', seed=50,
                        callbacks=callbacks)

    run_time = timer() - start

    # best score extraction
    # the try-except block was inserted because of two versions of LIGHTGBM is desktop and server. The server
    # version used keyword 'valid rmse-mean' while the desktop version was using 'rmse-mean'
    try:
        best_rmse = np.min(cv_results[
                               'valid rmse-mean'])  # valid rmse-mean stands for mean RMSE value across all the folds for each boosting round
    except:
        best_rmse = np.min(cv_results['rmse-mean'])

    # result of each iteration will be store in the iteration_csv
    if ITERATION == 1:
        makedirs([os.path.dirname(iteration_csv)])

        write_to = open(iteration_csv, 'w')
        writer = csv.writer(write_to)
        writer.writerows([['loss', 'params', 'iteration', 'run_time'],
                          [best_rmse, params, ITERATION, run_time]])
        write_to.close()

    else:  # when ITERATION > 0, will append result on the existing csv/file
        write_to = open(iteration_csv, 'a')
        writer = csv.writer(write_to)
        writer.writerow([best_rmse, params, ITERATION, run_time])

    # dictionary with information for evaluation
    return {'loss': best_rmse, 'params': params,
            'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}


def bayes_hyperparam_opt(x_train, y_train, iteration_csv, n_fold=10, max_evals=1000, skip_processing=False):
    """
    Hyperparameter optimization using Bayesian optimization method.

    *****
    good resources for building LGBM model

    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    https://neptune.ai/blog/lightgbm-parameters-guide

    Bayesian Hyperparameter Optimization:
    details at: https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f

    coding help from:
    1. https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
    2. https://www.kaggle.com/code/prashant111/bayesian-optimization-using-hyperopt
    *****

    :param x_train, y_train : Predictor and target arrays from split_train_test_ratio() function.
    :param n_fold : Number of folds in K Fold CV. Default set to 10.
    :param max_evals : Maximum number of evaluations during hyperparameter optimization. Default set to 1000.
    :param skip_processing: Set to True to skip hyperparameter tuning. Default set to False.

    :return : Best hyperparameters' dictionary.
    """
    if not skip_processing:
        print(f'performing bayesian hyperparameter optimization...')

        # merging x_train and y_train into a single dataset
        train_set = pd.concat([x_train, y_train], axis=1)

        # creating hyperparameter space for LGBM models
        param_space = {'boosting_type': hp.choice('boosting_type',
                                                  [{'boosting_type': 'gbdt',
                                                    'subsample': hp.uniform('gbdt_subsample', 0.5, 0.8)},
                                                   {'boosting_type': 'dart',
                                                    'subsample': hp.uniform('dart_subsample', 0.5, 0.8)},
                                                   {'boosting_type': 'goss', 'subsample': 1.0}]),
                       'n_estimators': hp.quniform('n_estimators', 100, 400, 25),
                       'max_depth': hp.uniform('max_depth', 5, 15),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                       'colsample_bynode': hp.uniform('colsample_bynode', 0.6, 1.0),
                       'path_smooth': hp.uniform('path_smooth', 0.1, 0.5),
                       'num_leaves': hp.quniform('num_leaves', 30, 70, 5),
                       'min_child_samples': hp.quniform('min_child_samples', 20, 50, 5)}

        # optimization algorithm
        tpe_algorithm = tpe.suggest  # stand for Tree-structured Parzen Estimator. A surrogate of the objective function.
        # the hyperparameter tuning approach, Sequential model-based optimization (SMBO), will
        # try to try to closely match the surrogate function to the objective function

        # keeping track of results
        bayes_trials = Trials()  # The Trials object will hold everything returned from the objective function in the

        # .results attribute. It also holds other information from the search, but we return
        # everything we need from the objective.

        # creating a wrapper function to bring all arguments of objective_func_bayes() under a single argument
        def objective_wrapper(params):
            return objective_func_bayes(params, train_set, iteration_csv, n_fold)

        # implementation of Sequential model-based optimization (SMBO)
        global ITERATION
        ITERATION = 0

        # run optimization
        best = fmin(fn=objective_wrapper, space=param_space, algo=tpe_algorithm,
                    max_evals=max_evals, trials=bayes_trials, rstate=np.random.default_rng(50))

        # sorting the trials to get the set of hyperparams with lowest loss
        bayes_trials_results = sorted(bayes_trials.results[1:],
                                      key=lambda x: x['loss'],
                                      reverse=False)  # the indexing in the results is done to remove {'status': 'new'} at 0 index
        best_hyperparams = bayes_trials_results[0]['params']

        print('\n')
        print('best hyperparameter set', '\n', best_hyperparams, '\n')
        print('best RMSE:', bayes_trials.results[1]['loss'])

        return best_hyperparams

    else:
        pass


def train_model(x_train, y_train, params_dict, n_jobs=-1,
                load_model=False, save_model=False, save_folder=None, model_save_name=None,
                skip_tune_hyperparameters=False, iteration_csv=None, n_fold=10, max_evals=1000):
    """
    Train a LightGBM regressor model with given hyperparameters.

    *******
    # To run the model without saving/loading the trained model, use load_model=False, save_model=False, save_folder=None,
        model_save_name=None.
    # To run the model and save it without loading any trained model, use load_model=False, save_model=True,
        save_folder='give a folder path', model_save_name='give a name'.
    # To load a pretrained model without running a new model, use load_model=True, save_model=False,
        save_folder='give the saved folder path', model_save_name='give the saved name'.
    *******

    :param x_train, y_train : x_train (predictor) and y_train (target) arrays from split_train_test_ratio() function.
    :param params_dict : ML model param dictionary. Currently supports LGBM model 'gbdt', 'goss', and 'dart'.
                  **** when tuning hyperparameters set params_dict=None.
                    For LGBM the dictionary should be like the following with user defined values-
                    param_dict = {'boosting_type': 'gbdt',
                                  'colsample_bynode': 0.7,
                                  'colsample_bytree': 0.8,
                                  'learning_rate': 0.05,
                                  'max_depth': 13,
                                  'min_child_samples': 40,
                                  'n_estimators': 250,
                                  'num_leaves': 70,
                                  'path_smooth': 0.2,
                                  'subsample': 0.7}
    :param n_jobs: The number of jobs to run in parallel. Default set to to -1 (using all processors).
    :param load_model : Set to True if want to load saved model. Default set to False.
    :param save_model : Set to True if want to save model. Default set to False.
    :param save_folder : Filepath of folder to save model. Default set to None for save_model=False..
    :param model_save_name : Model's name to save with. Default set to None for save_model=False.
    :param skip_tune_hyperparameters: Set to True to skip hyperparameter tuning. Default set to False.
    :param iteration_csv : Filepath of a csv where hyperparameter iteration step will be stored.
    :param n_fold : Number of folds in K Fold CV. Default set to 10.
    :param max_evals : Maximum number of evaluations during hyperparameter optimization. Default set to 1000.

    :return: trained LGBM regression model.
    """
    global reg_model

    if not load_model:
        print(f'Training model...')
        start_time = timeit.default_timer()
        if not skip_tune_hyperparameters:
            params_dict = bayes_hyperparam_opt(x_train, y_train, iteration_csv,
                                               n_fold=n_fold, max_evals=max_evals,
                                               skip_processing=skip_tune_hyperparameters)

        # Configuring the regressor with the parameters
        reg_model = LGBMRegressor(tree_learner='serial', random_state=0,
                                  deterministic=True, force_row_wise=True,
                                  n_jobs=n_jobs, **params_dict)

        trained_model = reg_model.fit(x_train, y_train)
        y_pred = trained_model.predict(x_train)

        print('Train RMSE = {:.3f}'.format(calculate_rmse(Y_pred=y_pred, Y_obsv=y_train)))
        print('Train R2 = {:.3f}'.format(calculate_r2(Y_pred=y_pred, Y_obsv=y_train)))

        if save_model:
            makedirs([save_folder])
            if '.joblib' not in model_save_name:
                model_save_name = model_save_name + '.joblib'

            save_path = os.path.join(save_folder, model_save_name)
            joblib.dump(trained_model, save_path, compress=3)

        # printing and saving runtime
        end_time = timeit.default_timer()
        runtime = (end_time - start_time) / 60
        run_str = f'model training time {runtime} mins'
        print('model training time {:.3f} mins'.format(runtime))

        if not skip_tune_hyperparameters:  # saving hyperparameter tuning + model training time
            runtime_save = os.path.join(save_folder, model_save_name + '_tuning_training_runtime.txt')
            with open(runtime_save, 'w') as file:
                file.write(run_str)

        else:  # saving model training time with given parameters
            runtime_save = os.path.join(save_folder, model_save_name + '_training_runtime.txt')
            with open(runtime_save, 'w') as file:
                file.write(run_str)

    else:
        print('Loading trained model...')

        if '.joblib' not in model_save_name:
            model_save_name = model_save_name + '.joblib'
        saved_model_path = os.path.join(save_folder, model_save_name)
        trained_model = joblib.load(saved_model_path)
        print('Loaded trained model.')

    return trained_model


def create_pdplots(trained_model, x_train, features_to_include, output_dir, plot_name,
                   ylabel='Effective Precipitation \n (mm)',
                   skip_processing=False):
    """
    Plot partial dependence plot.

    :param trained_model: Trained model object.
    :param x_train: x_train dataframe (if the model was trained with a x_train as dataframe) or array.
    :param features_to_include: List of features for which PDP plots will be made. If set to 'All', then PDP plot for
                                all input variables will be created.
    :param output_dir: Filepath of output directory to save the PDP plot.
    :param plot_name: str of plot name. Must include '.jpeg' or 'png'.
    :param ylabel: Ylabel for partial dependence plot. Default set to Effective Precipitation \n (mm)' for monthly model.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # creating variables for unit degree and degree celcius
        deg_unit = r'$^\circ$'
        deg_cel_unit = r'$^\circ$C'

        # plotting
        if features_to_include == 'All':  # to plot PDP for all attributes
            features_to_include = list(x_train.columns)

        plt.rcParams['font.size'] = 30

        pdisp = PDisp.from_estimator(trained_model, x_train, features=features_to_include,
                                     percentiles=(0.05, 1), subsample=0.8, grid_resolution=20,
                                     n_jobs=-1, random_state=0)

        # creating a dictionary to rename PDP plot labels
        feature_dict = {
            'GRIDMET_Precip': 'Precipitation (mm)', 'GRIDMET_Precip_1_lag': 'Precipitation lagged - 1 month (mm)',
            'GRIDMET_Precip_2_lag': 'Precipitation lagged - 2 month (mm)',
            'PRISM_Tmax': f'Max. Temperature ({deg_cel_unit})',
            'GRIDMET_RET': 'Reference ET (mm)', 'GRIDMET_vap_pres_def': 'Vapour pressure deficit (kpa)',
            'GRIDMET_max_RH': 'Max. relative humidity (%)', 'GRIDMET_min_RH': 'Min relative humidity (%)',
            'GRIDMET_wind_vel': 'Wind velocity (m/s)', 'GRIDMET_short_rad': 'Downward shortwave radiation (W/$m^2$)',
            'DAYMET_sun_hr': 'Daylight duration (hr)', 'Bulk_density': 'Bulk Density (kg/$m^3$)',
            'Clay_content': 'Clay content (%)', 'Field_capacity': 'Field Capacity (%)',
            'Sand_content': 'Sand Content (%)',
            'AWC': 'Available water capacity (mm)', 'DEM': 'Elevation', 'month': 'Month', 'Slope': 'Slope (%)',
            'Latitude': f'Latitude ({deg_unit})', 'Longitude': f'Longitude ({deg_unit})',
            'TERRACLIMATE_SR': 'Surface runoff (mm)',
            'Runoff_precip_fraction': 'Runoff-Precipitation fraction',
            'Precipitation_intensity': 'Precipitation intensity (mm/day)',
            'Dryness_index': 'RET/P', 'Relative_infiltration_capacity': 'Relative infiltration capacity',
            'PET_P_corr': 'RET-P seasonal correlation'
        }

        # Subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)',
                          '(l)', '(m)', '(n)', '(o)', '(p)']

        # replacing x and y axis labels
        row_num = range(0, pdisp.axes_.shape[0])
        col_num = range(0, pdisp.axes_.shape[1])

        feature_idx = 0
        for r in row_num:
            for c in col_num:
                if pdisp.axes_[r][c] is not None:
                    pdisp.axes_[r][c].set_xlabel(feature_dict[features_to_include[feature_idx]])

                    # subplot num
                    pdisp.axes_[r][c].text(0.1, 0.9, subplot_labels[feature_idx], transform=pdisp.axes_[r][c].transAxes,
                                           fontsize=35, va='top', ha='left')

                    feature_idx += 1
                else:
                    pass

        for row_idx in range(0, pdisp.axes_.shape[0]):
            pdisp.axes_[row_idx][0].set_ylabel(ylabel)

        fig = plt.gcf()
        fig.set_size_inches(30, 30)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')

        print('PDP plots generated...')

    else:
        pass


def create_aleplots(trained_model, x_train, y_train, features_to_include,
                    output_dir, plot_name, make_CI=True, skip_processing=False):
    """
    Plot Accumulated Local Effects (ALE) plot.

    :param trained_model: Trained model object.
    :param x_train: x_train dataframe (if the model was trained with a x_train as dataframe) or array.
    :param y_train: y_train dataframe (if the model was trained with a x_train as dataframe) or array.
    :param features_to_include: List of features for which ALE plots will be made.
                                If set to 'All', then PDP plot for all input variables will be created.
    :param output_dir: Filepath of output directory to save the PDP plot.
    :param plot_name: str of plot name. Must include '.jpeg' or 'png'.
    :param make_CI: Set to True if want to include CI in the ALE plot. The confidence intervals are simply the uncertainty
               in the mean value. This function uses 100 bootstraping to estimate the CIs.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # creating variables for unit degree and degree celcius
        deg_unit = r'$^\circ$'
        deg_cel_unit = r'$^\circ$C'

        # plotting
        if features_to_include == 'All':  # to plot PDP for all attributes
            features_to_include = list(x_train.columns)

        # creating a dictionary to rename PDP plot labels
        feature_dict = {
            'GRIDMET_Precip': 'Precipitation (mm)', 'GRIDMET_Precip_1_lag': 'Precipitation lagged -\n 1 month (mm)',
            'GRIDMET_Precip_2_lag': 'Precipitation lagged -\n 2 month (mm)',
            'PRISM_Tmax': f'Max. Temperature ({deg_cel_unit})',
            'GRIDMET_RET': 'Reference ET (mm)', 'GRIDMET_vap_pres_def': 'Vapour pressure deficit (kpa)',
            'GRIDMET_max_RH': 'Max. relative humidity (%)', 'GRIDMET_min_RH': 'Min relative humidity (%)',
            'GRIDMET_wind_vel': 'Wind velocity (m/s)', 'GRIDMET_short_rad': 'Downward shortwave \n radiation (W/$m^2$)',
            'DAYMET_sun_hr': 'Daylight duration (hr)', 'Bulk_density': 'Bulk Density (kg/$m^3$)',
            'Clay_content': 'Clay content (%)', 'Field_capacity': 'Field Capacity (%)',
            'Sand_content': 'Sand Content (%)',
            'AWC': 'Available water \n capacity (mm)', 'DEM': 'Elevation', 'month': 'Month', 'Slope': 'Slope (%)',
            'Latitude': f'Latitude ({deg_unit})', 'Longitude': f'Longitude ({deg_unit})',
            'TERRACLIMATE_SR': 'Surface runoff (mm)',
            'Runoff_precip_fraction': 'Runoff-Precipitation fraction',
            'Precipitation_intensity': 'Precipitation intensity \n (mm/day)',
            'Dryness_index': 'RET/P', 'Relative_infiltration_capacity': 'Relative infiltration capacity',
            'PET_P_corr': 'RET-P seasonal \n correlation'
        }

        plt.rcParams['font.size'] = 8

        # creating explainer object and calculating 1d ale
        if make_CI:
            bootstrap = 100
        else:
            bootstrap = 1

        explainer = skexplain.ExplainToolkit(('LGBMRegressor', trained_model), X=x_train, y=y_train)
        ale_1d_ds = explainer.ale(features=features_to_include, n_bootstrap=bootstrap,
                                  subsample=50000, n_jobs=10, n_bins=20)

        # Create ALE plots
        fig, axes = explainer.plot_ale(
            ale=ale_1d_ds,
            features=features_to_include,  # Important features you want to plot
            display_feature_names=feature_dict,  # Feature names
            figsize=(10, 8)
        )

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(os.path.join(output_dir, plot_name))

        print('ALE plots generated...')

    else:
        pass


def plot_permutation_importance(trained_model, x_test, y_test, output_dir, plot_name,
                                saved_var_list_name,
                                exclude_columns=None, skip_processing=False):
    """
    Plot permutation importance for model predictors.

    :param trained_model: Trained ML model object.
    :param x_test: Filepath of x_test csv or dataframe. In case of dataframe, it has to come directly from the
                    split_train_val_test_set() function.
    :param y_test: Filepath of y_test csv or dataframe.
    :param exclude_columns: List of predictors to be excluded.
                            Exclude the same predictors for which model wasn't trained. In case the x_test comes as a
                            dataframe from the split_train_val_test_set() function, set exclude_columns to None.
    :param output_dir: Output directory filepath to save the plot.
    :param plot_name: Plot name. Must contain 'png', 'jpeg'.
    :param saved_var_list_name: The name of to use to save the sorted important vars list. Must contain 'pkl'.
    :param skip_processing: Set to True to skip this process.

    :return: List of sorted (most important to less important) important variable names.
    """
    if not skip_processing:
        makedirs([output_dir])

        if '.csv' in x_test:
            # Loading x_test and y_test
            x_test_df = pd.read_csv(x_test)
            x_test_df = x_test_df.drop(columns=exclude_columns)
            x_test_df = reindex_df(x_test_df)

            y_test_df = pd.read_csv(y_test)
        else:
            x_test_df = x_test
            y_test_df = y_test

        # ensure arrays are writable  (the numpy conversion code block was added after a conda env upgrade threw 'WRITABLE array'
        #                              error, took chatgpt's help to figure this out. The error meant - permutation_importance() was
        #                              trying to change the array but could not as it was writable before. This code black makes the
        #                              arrays writable)
        x_test_np = x_test_df.to_numpy()
        y_test_np = y_test_df.to_numpy()

        x_test_np.setflags(write=True)
        y_test_np.setflags(write=True)

        # generating permutation importance score on test set
        result_test = permutation_importance(trained_model, x_test_np, y_test_np,
                                             n_repeats=30, random_state=0, n_jobs=-1, scoring='r2')

        sorted_importances_idx = result_test.importances_mean.argsort()
        predictor_cols = x_test_df.columns
        importances = pd.DataFrame(result_test.importances[sorted_importances_idx].T,
                                   columns=predictor_cols[sorted_importances_idx])

        # sorted important variables
        sorted_imp_vars = importances.columns.tolist()[::-1]
        print('\n', 'Sorted Important Variables:', sorted_imp_vars, '\n')

        # renaming predictor names
        rename_dict = {'GRIDMET_Precip': 'Precipitation', 'GRIDMET_Precip_1_lag': 'Precipitation lagged - 1 month',
                       'GRIDMET_Precip_2_lag': 'Precipitation lagged - 2 month', 'GRIDMET_RET': 'Reference ET',
                       'GRIDMET_vap_pres_def': 'Vapor pressure deficit', 'GRIDMET_max_RH': 'Max. relative humidity',
                       'GRIDMET_short_rad': 'Downward shortwave radiation', 'DAYMET_sun_hr': 'Daylight duration',
                       'Field_capacity': 'Field capacity', 'Sand_content': 'Sand content',
                       'AWC': 'Available water capacity', 'DEM': 'Elevation', 'month': 'Month',
                       'PRISM_Tmax': f'Max. temperature', 'TERRACLIMATE_SR': 'Surface runoff',
                       'Runoff_precip_fraction': 'Runoff-Precipitation fraction',
                       'Precipitation_intensity': 'Precipitation intensity',
                       'Relative_infiltration_capacity': 'Relative infiltration capacity',
                       'Dryness_index': 'RET/P', 'PET_P_corr': 'RET-P seasonal correlation',
                       'Clay_content': 'Clay content'}

        importances = importances.rename(columns=rename_dict)

        # plotting
        plt.figure(figsize=(6, 4))

        ax = importances.plot.box(vert=False, whis=10)
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_xlabel('Relative change in accuracy', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_name), dpi=200)

        # saving the list to avoid running the permutation importance plot if not required (saves model running time)
        joblib.dump(sorted_imp_vars, os.path.join(output_dir, saved_var_list_name))

        print('Permutation importance plot generated...')

    else:
        sorted_imp_vars = joblib.load(os.path.join(output_dir, saved_var_list_name))

    return sorted_imp_vars
