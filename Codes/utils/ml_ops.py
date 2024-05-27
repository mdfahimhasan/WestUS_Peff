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
from timeit import default_timer as timer

import lightgbm as lgb
from lightgbm import LGBMRegressor

from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

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


def create_train_test_dataframe(years_list, month_range, monthly_data_path_dict, yearly_data_path_dict,
                                static_data_path_dict, datasets_to_include, output_parquet,
                                skip_processing=False, n_partitions=20):
    """
    Compile monthly/yearly/static datasets into a dataframe. This function-generated dataframe will be used as
    train-test data for ML models.

    *** if there is no monthly dataset, set month_range and monthly_data_path_dict to None.
    *** if there is no yearly dataset, set yearly_data_path_dict to None.
    *** if there is no static data, set static_data_path_dict to None.

    :param years_list: A list of year_list for which data to include in the dataframe.
    :param month_range: A tuple of start and end month for which data to filter. Set to None if there is no monthly dataset.
    :param monthly_data_path_dict: A dictionary with monthly variables' names as keys and their paths as values.
                                   Set to None if there is no monthly dataset.
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
        output_dir = os.path.dirname(output_parquet)
        makedirs([output_dir])

        variable_dict = {}
        yearly_month_count_dict = {}

        if month_range is not None:
            month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
        else:
            month_list = None

        if monthly_data_path_dict is not None:
            for var in monthly_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}...')

                    for year in years_list:
                        for month_count, month in enumerate(month_list):
                            monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]

                            data_arr = read_raster_arr_object(monthly_data, get_file=False).flatten()
                            len_arr = len(list(data_arr))
                            year_data = [int(year)] * len_arr
                            month_data = [int(month)] * len_arr

                            if (month_count == 0) & (var not in variable_dict.keys()):
                                variable_dict[var] = list(data_arr)
                                variable_dict['year'] = year_data
                                variable_dict['month'] = month_data
                            else:
                                variable_dict[var].extend(list(data_arr))
                                variable_dict['year'].extend(year_data)
                                variable_dict['month'].extend(month_data)

                            yearly_month_count_dict[year] = month_count + 1

        if yearly_data_path_dict is not None:
            for var in yearly_data_path_dict.keys():
                if var in datasets_to_include:
                    print(f'processing data for {var}..')

                    for year_count, year in enumerate(years_list):
                        yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]

                        data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                        if (year_count == 0) & (var not in variable_dict.keys()):
                            if monthly_data_path_dict is not None:
                                variable_dict[var] = list(data_arr) * yearly_month_count_dict[year]
                            else:  # if no monthly data
                                variable_dict[var] = list(data_arr)

                        else:
                            if monthly_data_path_dict is not None:
                                variable_dict[var].extend(list(data_arr) * yearly_month_count_dict[year])
                            else:  # if no monthly data
                                variable_dict[var].extend(list(data_arr))

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

        if 'Rainfed_Frac' in train_test_ddf.columns:  # renaming 'Rainfed_Frac' as this attribute will not be
                                                      # available for Irrigated pixels
            train_test_ddf = train_test_ddf.rename(columns={'Rainfed_Frac': 'Cropland_Frac'})

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
                             random_state=0, verbose=True,  remove_outlier=False,
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
        input_df = pd.read_parquet(input_csv)

        if month_range is not None: # filter for specific month ranges
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
    :param years_in_train: List of year_list to keep as train dataset. Input multiple year_list.
    :param year_in_test: List of year to keep as test dataset. Input single year.
    :param output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    :param verbose : Set to True if want to print which columns are being dropped and which will be included
                     in the model.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    if not skip_processing:
        print(f'Making train-test split with...', '\n',
              f'year_list {years_in_train} in train set', '\n',
              f'year {year_in_test} in test set')

        input_df = pd.read_parquet(input_csv)
        drop_columns = exclude_columns + [
            pred_attr]  # dropping unwanted columns/columns that will not be used in model training

        # making train-test split based on provided year_list
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
        best_rmse = np.min(cv_results['valid rmse-mean'])  # valid rmse-mean stands for mean RMSE value across all the folds for each boosting round
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


def bayes_hyperparam_opt(x_train, y_train, iteration_csv,  n_fold=10, max_evals=1000, skip_processing=False):
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
                                                  [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gbdt_subsample', 0.5, 0.8)},
                                                   {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 0.8)},
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


def create_pdplots(trained_model, x_train, features_to_include, output_dir, plot_name, skip_processing=False):
    """

    :param trained_model: Trained model object.
    :param x_train: x_train dataframe (if the model was trained with a x_train as dataframe) or array.
    :param features_to_include: List of features for which PDP plots will be made. If set to 'All', then PDP plot for
                                all input variables will be created.
    :param output_dir: Filepath of output directory to save the PDP plot.
    :param plot_name: str of plot name. Must include '.jpeg' or 'png'.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        # creating variables for unit degree and degree celcius
        deg_unit = r'$^\circ$'
        deg_cel_unit = r'$^\circ$C'

        # creating a dictionary to rename input variables in the PDP plot
        feature_dict = {
            'Cropland_Frac': '% Cropland',
            'MODIS_Day_LST': 'MODIS Land Surface Temp (K)', 'MODIS_LAI': 'MODIS LAI',
            'MODIS_NDVI': 'MODIS NDVI', 'MODIS_NDWI': 'MODIS NDWI', 'PRISM_Precip': 'PRISM Precipitation (mm)',
            'PRISM_Tmax': f'PRISM Max Temp ({deg_cel_unit})', 'PRISM_Tmin': f'PRISM Min Temp ({deg_cel_unit})',
            'Ssebop_ETa': 'SseBop ET (mm)', 'GRIDMET_Precip': 'GRIDMET Precipitation (mm)',
            'GRIDMET_RET': 'GRIDMET Reference ET (mm)', 'GRIDMET_vap_pres_def': 'GRIDMET Mean Vapour Pressure Deficit (kpa)',
            'GRIDMET_max_RH': 'GRIDMET Max Relative Humidity (%)', 'GRIDMET_min_RH': 'GRIDMET Min Relative Humidity (%)',
            'GRIDMET_wind_vel': 'GRIDMET Wind Velocity (m/s)', 'GRIDMET_short_rad': 'GRIDMET Downward Shortwave Radiation (W/m^2)',
            'DAYMET_sun_hr': 'DAYMET Daylight Duration (hr)', 'Bulk_density': 'Bulk Density (kg/m^3)',
            'Clay_content': 'Clay Content (%)', 'Field_capacity': 'Field Capacity (%)', 'Sand_content': 'Sand Content (%)',
            'Slope': 'Slope (%)', 'AWC': 'Available Water Capacity (mm)',
            'Latitude': f'Latitude ({deg_unit})', 'Longitude': f'Longitude ({deg_unit})'
        }

        # renaming columns
        x_train = x_train.rename(columns=feature_dict)

        # plotting
        if features_to_include == 'All':  # to plot PDP for all attributes
            features_to_include = list(x_train.columns)

        pdisp = PDisp.from_estimator(trained_model, x_train, features=features_to_include,
                                     percentiles=(0.05, 1), subsample=0.8, grid_resolution=20,
                                     n_jobs=-1, random_state=0)
        # replacing Y axis labels
        for row_idx in range(0, pdisp.axes_.shape[0]):
            pdisp.axes_[row_idx][0].set_ylabel('Effective Precipitation')

        fig = plt.gcf()
        plt.rcParams['font.size'] = 20  # doesn't work
        fig.set_size_inches(30, 25)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(os.path.join(output_dir, plot_name), dpi=100, bbox_inches='tight')
    else:
        pass


def plot_permutation_importance(trained_model, x_test, y_test, output_dir, plot_name,
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
    :param skip_processing: Set to True to skip this process.

    :return: None
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

        # renaming predictor names
        rename_dict = {'GRIDMET_Precip' : 'Precipitation', 'GRIDMET_RET': 'Reference ET',
                       'GRIDMET_vap_pres_def': 'Vapor pressure deficit', 'GRIDMET_max_RH': 'Max. relative humidity',
                       'GRIDMET_short_rad': 'Downward shortwave radiation', 'DAYMET_sun_hr': 'Daylight duration',
                       'Field_capacity': 'Field capacity', 'Sand_content': 'Sand content',
                       'AWC': 'Available water capacity', 'DEM': 'Elevation', 'month': 'Month'}

        importances = importances.rename(columns=rename_dict)

        # plotting
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': 8})

        ax = importances.plot.box(vert=False, whis=10)
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_xlabel('Relative change in accuracy')
        ax.figure.tight_layout()

        plt.savefig(os.path.join(output_dir, plot_name), dpi=200)
    else:
        pass