import os
import sys
import joblib
import timeit
import pandas as pd
from glob import glob
from pprint import pprint
import dask.dataframe as ddf
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, RepeatedKFold
from sklearn.inspection import PartialDependenceDisplay as PDisp

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object
from Codes.utils.stats_ops import calculate_r2, calculate_rmse
from Codes.utils.plots import scatter_plot_of_same_vars, density_grid_plot_of_same_vars

no_data_value = -9999
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
model_res = 0.02000000000000000736  # in deg, ~2.22 km


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


def create_monthly_train_test_dataframe(years_list, month_range, monthly_data_path_dict, yearly_data_path_dict,
                                        static_data_path_dict, datasets_to_include, output_parquet,
                                        skip_processing=False, n_partitions=20):
    """

    :param years_list: A list of years for which data to include in the dataframe.
    :param month_range: A tuple of start and end month for which data to filter. Default set to None.
    :param monthly_data_path_dict: A dictionary with monthly variables' names as keys and their paths as values.
    :param yearly_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
    :param static_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_parquet: Output filepath of the parquet file to save. Using parquet as it requres lesser memory.
    :param skip_processing: Set to True to skip this dataframe creation process.
    :param n_partitions: Number of partitions to save the parquet file in using dask dataframe.

    :return: The filepath of the output parquet file.
    """
    if not skip_processing:
        output_dir = os.path.dirname(output_parquet)
        makedirs([output_dir])

        variable_dict = {}
        yearly_month_count_dict = {}

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

        for var in monthly_data_path_dict.keys():
            if var in datasets_to_include:
                print(f'processing data for {var}..')

                for year in years_list:
                    for month_count, month in enumerate(month_list):
                        monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}*.tif'))[0]

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

        for var in yearly_data_path_dict.keys():
            if var in datasets_to_include:
                print(f'processing data for {var}..')

                for year_count, year in enumerate(years_list):
                    yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]

                    data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()
                    data_duplicated_for_months = list(data_arr) * yearly_month_count_dict[year]

                    if (year_count == 0) & (var not in variable_dict.keys()):
                        variable_dict[var] = data_duplicated_for_months

                    else:
                        variable_dict[var].extend(data_duplicated_for_months)

        total_month_count = 0
        for i in yearly_month_count_dict.values():
            total_month_count += i

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

        train_test_ddf.to_parquet(output_parquet, write_index=False)

        return output_parquet

    else:
        return output_parquet


def split_train_val_test_set(input_csv, pred_attr, exclude_columns, output_dir, model_version,
                             month_range=None, test_perc=0.3, validation_perc=0,
                             random_state=0, verbose=True, skip_processing=False):
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
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    global x_val, y_val

    if not skip_processing:
        input_df = pd.read_parquet(input_csv)

        if month_range is not None: # filter for specific month ranges
            month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
            input_df = input_df[input_df['month'].isin(month_list)]

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
    :param years_in_train: List of years to keep as train dataset. Input multiple years.
    :param year_in_test: List of year to keep as test dataset. Input single year.
    :param output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    :param verbose : Set to True if want to print which columns are being dropped and which will be included
                     in the model.
    :param skip_processing: Set to True if want to skip merging IrrMapper and LANID extent data patches.

    returns: X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    if not skip_processing:
        print(f'Making train-test split with...', '\n',
              f'years {years_in_train} in train set', '\n',
              f'year {year_in_test} in test set')

        input_df = pd.read_parquet(input_csv)
        drop_columns = exclude_columns + [
            pred_attr]  # dropping unwanted columns/columns that will not be used in model training

        # making train-test split based on provided years
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


def tune_hyperparameter(x, y, model='rf', n_folds=5, repeated_Kfold=False, n_repeats=5,
                        random_search=True, n_iter=50, n_jobs=-1):
    """
    Hyperparameter optimization using RandomizedSearchCV/GridSearchCV.

    *****
    good resources for building LGBM model

    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    https://neptune.ai/blog/lightgbm-parameters-guide
    *****

    Parameters:
    x_val, y_val : x_val (predictor) and y_val (target) arrays from split_train_test_ratio function.
    model : Model for which hyperparameters will be tuned. Can only tune hyperparameters for RF regressor now.
            Default set to 'rf'.
    n_folds : Number of folds in K Fold CV. Default set to 5.
    repeated_Kfold : Set to True if want to perform repeated Kfold. If False (default), will run for KFold.
    n_repeats : If repeated_Kfold is True, number of repeats. Default set to 5.
    random_search : Set to False if want to perform GridSearchCV. Default set to True to perform RandomizedSearchCV.
    n_iter : Number of parameter combinations to be tested in RandomizedSearchCV if random_search is True.
    n_jobs (rf/gbdt param): The number of jobs to run in parallel. Defaults to -1(using all processors).

    Returns : Optimized Hyperparameters.
    """
    global regressor

    print(f'Finding optimal hyperparameters...')

    # creating parameter dictionary
    # hyperparameters are optimized from param_to_optimize_dict
    # if hyperparamter optimization is off/not needed uses paramters from default_params_dict
    # ******* after hyperparamter optimization, assign optimized values to default_params_dict***************
    param_to_optimize_dict = {'rf': {'n_estimators': [100, 200, 300, 400, 500],
                                     'max_depth': [7, 10, 15, 20],
                                     'max_features': [6, 7, 10, 'log2'],
                                     'min_samples_leaf': [5e-4, 1e-5, 1e-3, 6, 12, 20, 25],
                                     'min_samples_split': [6, 7, 8, 10],
                                     'max_samples': [None, 0.9, 0.8, 0.7]
                                     },
                              'lgbm': {'n_estimators': [100, 200, 250],
                                       'max_depth': [7, 10, 13],
                                       'learning_rate': [0.01, 0.05],
                                       'subsample': [0.8, 0.7, 0.6],
                                       'colsample_bytree': [0.8, 0.7],
                                       'colsample_bynode': [0.8, 0.7],
                                       'path_smooth': [0.1, 0.2, 0.3],
                                       'num_leaves': [30, 50, 70],
                                       'min_child_samples': [20, 25, 40],
                                       #                                    'data_sample_strategy' : ['goss']
                                       }
                              }
    param_dict = param_to_optimize_dict
    print('Model Name:', model)
    pprint(param_dict[model])

    # creating model structures
    if model == 'rf':
        regressor = RandomForestRegressor(random_state=0, n_jobs=n_jobs, bootstrap=True, oob_score=True)

    elif model == 'lgbm':
        # the boosting_type has been set to 'goss' for faster training. Can use 'gdbt'/'dart'. Change params_dict accordingly
        regressor = LGBMRegressor(tree_learner='serial', random_state=0,
                                  deterministic=True, force_row_wise=True, n_jobs=n_jobs)
    else:
        raise Exception("model should be 'rf'/'lgbm'. Other types are not supported currently")

    scoring_metrics = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']

    # Hyperparameter optimization block
    # KFold or repeated KFold
    if repeated_Kfold:
        kfold = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=0)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Random search or grid search
    if random_search:
        fitted_model = RandomizedSearchCV(estimator=regressor, param_distributions=param_dict[model], n_iter=n_iter,
                                          cv=kfold, verbose=1, random_state=0, n_jobs=n_jobs,
                                          scoring=scoring_metrics, refit=scoring_metrics[1], return_train_score=True)
    else:
        fitted_model = GridSearchCV(estimator=regressor, param_grid=param_dict[model], cv=kfold, verbose=1,
                                    n_jobs=n_jobs,
                                    scoring=scoring_metrics, refit=scoring_metrics[1], return_train_score=True)

    fitted_model.fit(x, y)  # this will be x_val and y_val if tune_hyperparameter=True

    print('\n')
    print('best parameters for RMSE value', '\n')
    pprint(fitted_model.best_params_)
    print('\n')
    print('Train Results....')
    best_rmse = fitted_model.cv_results_['mean_train_neg_root_mean_squared_error'][fitted_model.best_index_]
    best_r2 = fitted_model.cv_results_['mean_train_r2'][fitted_model.best_index_]
    best_MAE = fitted_model.cv_results_['mean_train_neg_mean_absolute_error'][fitted_model.best_index_]
    print('Best tuning-train RMSE: {:.3f}'.format(best_rmse))
    print('Best tuning-train R2: {:.3f}'.format(best_r2))
    print('Best tuning-train MAE: {:.3f}'.format(best_MAE))

    print('\n')
    print('Test Results....')
    best_rmse = fitted_model.cv_results_['mean_test_neg_root_mean_squared_error'][fitted_model.best_index_]
    best_r2 = fitted_model.cv_results_['mean_test_r2'][fitted_model.best_index_]
    best_MAE = fitted_model.cv_results_['mean_test_neg_mean_absolute_error'][fitted_model.best_index_]
    print('Best tuning-test RMSE: {:.3f}'.format(best_rmse))
    print('Best tuning-test R2: {:.3f}'.format(best_r2))
    print('Best tuning-test MAE: {:.3f}'.format(best_MAE))

    if model == 'rf':
        param_dict = {'n_estimators': fitted_model.best_params_['n_estimators'],
                      'max_depth': fitted_model.best_params_['max_depth'],
                      'max_features': fitted_model.best_params_['max_features'],
                      'min_samples_leaf': fitted_model.best_params_['min_samples_leaf'],
                      'min_samples_split': fitted_model.best_params_['min_samples_split'],
                      'max_samples': fitted_model.best_params_['max_samples']
                      }
    elif model == 'lgbm':
        param_dict = {'n_estimators': fitted_model.best_params_['n_estimators'],
                      'max_depth': fitted_model.best_params_['max_depth'],
                      'learning_rate': fitted_model.best_params_['learning_rate'],
                      'subsample': fitted_model.best_params_['subsample'],
                      'colsample_bytree': fitted_model.best_params_['colsample_bytree'],
                      'colsample_bynode': fitted_model.best_params_['colsample_bynode'],
                      'path_smooth': fitted_model.best_params_['path_smooth'],
                      'num_leaves': fitted_model.best_params_['num_leaves'],
                      'min_child_samples': fitted_model.best_params_['min_child_samples'],
                      #                       'data_sample_strategy' : fitted_model.best_params_['data_sample_strategy']
                      }

    return param_dict


def train_model(x_train, y_train, params_dict, model='rf', n_jobs=-1,
                load_model=False, save_model=False, save_folder=None, model_save_name=None,
                tune_hyperparameters=False, repeated_Kfold=False, n_folds=5, n_iter=10, n_repeats=5):
    """
    Train a Random Forest Regressor model with given hyperparameters.


    *******
    # To run the model without saving/loading the trained model, use load_model=False, save_model=False, save_folder=None,
        model_save_name=None.
    # To run the model and save it without loading any trained model, use load_model=False, save_model=True,
        save_folder='give a folder path', model_save_name='give a name'.
    # To load a pretrained model without running a new model, use load_model=True, save_model=False,
        save_folder='give the saved folder path', model_save_name='give the saved name'.
    *******


    params:
    x_train, y_train : x_train (predictor) and y_train (target) arrays from split_train_test_ratio function.
    model : str of type of model. The code can only run random forest regession model. Default set to 'rf'.
    params_dict : ML model param dictionary. Currently supports 'random forest (RF)' and 'LGBM (lgbm)' Goss.
                  **** when tuning hyperparameters set params_dict=None.
                  For RF the dictionary should be like the folowing with user defined values-
                    param_dict = {'n_estimators': 200,
                                  'max_depth': 8,
                                  'max_features': 'log2',
                                  'min_samples_leaf': 6,
                                  'min_samples_split': 4,
                                  'max_samples': None
                                 }
                For LGBM the dictionary shoudl be like the folowing with user defined values-
                    param_dict = {'n_estimators': 250,
                                  'max_depth': 13,
                                  'learning_rate': 0.05,
                                  'subsample': 0.7,
                                  'colsample_bytree': 0.8,
                                  'colsample_bynode': 0.7 ,
                                  'path_smooth': 0.2,
                                  'num_leaves': 70,
                                  'min_child_samples': 40,
                                  'data_sample_strategy' : 'goss'
                                  }

    n_jobs (rf/lgbm param): The number of jobs to run in parallel. Default set to to -1 (using all processors).
    load_model : Set to True if want to load saved model. Default set to False.
    save_model : Set to True if want to save model. Default set to False.
    save_folder : Filepath of folder to save model. Default set to None for save_model=False..
    model_save_name : Model's name to save with. Default set to None for save_model=False.

    returns: trained RF regression model.
    """
    global regressor_model

    if not load_model:
        print(f'Training {model} mode...')
        start_time = timeit.default_timer()
        if tune_hyperparameters:
            params_dict = tune_hyperparameter(x=x_train, y=y_train, model=model,
                                              n_folds=n_folds, repeated_Kfold=repeated_Kfold,
                                              n_repeats=n_repeats, n_iter=n_iter,
                                              random_search=True, n_jobs=n_jobs)

        if model == 'rf':
            n_estimators = params_dict['n_estimators']
            max_depth = params_dict['max_depth']
            max_features = params_dict['max_features']
            min_samples_leaf = params_dict['min_samples_leaf']
            min_samples_split = params_dict['min_samples_split']
            max_samples = params_dict['max_samples']
            regressor_model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                                                    max_depth=max_depth,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_samples_split=min_samples_split,
                                                    max_samples=max_samples, random_state=0, n_jobs=n_jobs,
                                                    bootstrap=True,
                                                    oob_score=True)
        elif model == 'lgbm':
            n_estimators = params_dict['n_estimators']
            max_depth = params_dict['max_depth']
            learning_rate = params_dict['learning_rate']
            subsample = params_dict['subsample']
            colsample_bytree = params_dict['colsample_bytree']
            colsample_bynode = params_dict['colsample_bynode']
            path_smooth = params_dict['path_smooth']
            num_leaves = params_dict['num_leaves']
            min_child_samples = params_dict['min_child_samples']

            # data_sample_strategy = 'goss'  # using 'goss' by default here as we are using 'GOSS' Gradient boosting methods

            # Configuring the regressor with the parameters
            regressor_model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                            subsample=subsample, colsample_bytree=colsample_bytree,
                                            colsample_bynode=colsample_bynode, path_smooth=path_smooth,
                                            num_leaves=num_leaves,
                                            min_child_samples=min_child_samples,
                                            # data_sample_strategy=data_sample_strategy,
                                            tree_learner='serial', random_state=0,
                                            deterministic=True, force_row_wise=True, n_jobs=n_jobs)

        trained_model = regressor_model.fit(x_train, y_train)
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

        if tune_hyperparameters:  # saving hyperparameter tuning + model training time
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


monthly_data_path_dict = {'Irrigated_cropET': '../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly',
                          'Rainfed_cropET': '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly',
                          'MODIS_Day_LST': '../../Data_main/Raster_data/MODIS_Day_LST/WestUS_monthly',
                          'MODIS_LAI': '../../Data_main/Raster_data/MODIS_LAI/WestUS_monthly',
                          'MODIS_NDVI': '../../Data_main/Raster_data/MODIS_NDVI/WestUS_monthly',
                          'MODIS_NDWI': '../../Data_main/Raster_data/MODIS_NDWI/WestUS_monthly',
                          'PRISM_Precip': '../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly',
                          'PRISM_Tmax': '../../Data_main/Raster_data/PRISM_Tmax/WestUS_monthly',
                          'PRISM_Tmin': '../../Data_main/Raster_data/PRISM_Tmin/WestUS_monthly',
                          'Ssebop_ETa': '../../Data_main/Raster_data/Ssebop_ETa/WestUS_monthly',
                          'GRIDMET_Precip': '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_monthly',
                          'GRIDMET_RET': '../../Data_main/Raster_data/Ssebop_ETa/WestUS_monthly',
                          'GRIDMET_vap_pres_def': '../../Data_main/Raster_data/GRIDMET_vap_pres_def/WestUS_monthly',
                          'GRIDMET_max_RH': '../../Data_main/Raster_data/GRIDMET_max_RH/WestUS_monthly',
                          'GRIDMET_min_RH': '../../Data_main/Raster_data/GRIDMET_min_RH/WestUS_monthly',
                          'GRIDMET_wind_vel': '../../Data_main/Raster_data/GRIDMET_wind_vel/WestUS_monthly',
                          'GRIDMET_short_rad': '../../Data_main/Raster_data/GRIDMET_short_rad/WestUS_monthly',
                          'DAYMET_sun_hr': '../../Data_main/Raster_data/DAYMET_sun_hr/WestUS_monthly'}

yearly_data_path_dict = {'Irrigated_cropland': '../../Data_main/Raster_data/Irrigated_cropland',
                         'Rainfed_cropland': '../../Data_main/Raster_data/Rainfed_cropland',
                         'Irrigated_Frac': '../../Data_main/Raster_data/Irrigated_cropland/Rainfed_Frac',
                         'Rainfed_Frac': '../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac'}

static_data_path_dict = {'Bulk_density': '../../Data_main/Raster_data/Bulk_density/WestUS',
                         'Clay_content': '../../Data_main/Raster_data/Clay_content/WestUS',
                         'Field_capacity': '../../Data_main/Raster_data/Field_capacity/WestUS',
                         'Organic_carbon_content': '../../Data_main/Raster_data/Organic_carbon_content/WestUS',
                         'Sand_content': '../../Data_main/Raster_data/Sand_content/WestUS',
                         'Slope': '../../Data_main/Raster_data/Slope',
                         'Latitude': '../../Data_main/Raster_data/Latitude',
                         'Longitude': '../../Data_main/Raster_data/Longitude'}

train_test_years_list = [2016, 2017, 2018, 2019, 2020]
total_month_range = (1, 12)  # considering March-November as growing season

datasets_to_include = ['Rainfed_cropET', 'Rainfed_Frac',
                       'MODIS_Day_LST', 'MODIS_LAI', 'MODIS_NDVI', 'MODIS_NDWI',
                       'PRISM_Precip', 'PRISM_Tmax', 'PRISM_Tmin', 'Ssebop_ETa',
                       'GRIDMET_Precip', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'GRIDMET_max_RH',
                       'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                       'Bulk_density', 'Clay_content', 'Field_capacity', 'Sand_content',
                       'Slope', 'Latitude', 'Longitude']

skip_train_test_df_creation = True
train_test_parquet_path = '../../Model_Run/Model_csv/train_test.parquet'

makedirs([os.path.dirname(train_test_parquet_path)])

create_monthly_train_test_dataframe(years_list=train_test_years_list,
                                    month_range=total_month_range,
                                    monthly_data_path_dict=monthly_data_path_dict,
                                    yearly_data_path_dict=yearly_data_path_dict,
                                    static_data_path_dict=static_data_path_dict,
                                    datasets_to_include=datasets_to_include,
                                    output_parquet=train_test_parquet_path,
                                    skip_processing=skip_train_test_df_creation)

output_dir = '../../Model_Run/Model_csv'
makedirs([output_dir])

model_version = 'v2'
train_test_month_range = (3, 11)
skip_train_test_split = False
exclude_columns = ['year', 'month', 'Ssebop_ETa', 'PRISM_Precip', 'GRIDMET_max_RH', 'GRIDMET_min_RH',
                   'GRIDMET_wind_vel', 'MODIS_NDWI', 'PRISM_Tmax', 'Latitude',
                   'Bulk_density', 'Field_capacity', 'Clay_content', 'Slope']


x_train, x_test, y_train, y_test = \
    split_train_val_test_set(input_csv=train_test_parquet_path, month_range=train_test_month_range,
                             model_version=model_version,
                             pred_attr='Rainfed_cropET', exclude_columns=exclude_columns,
                             output_dir=output_dir, test_perc=0.3, validation_perc=0,
                             random_state=0, verbose=True,
                             skip_processing=skip_train_test_split)

lgbm_param_dict = {'n_estimators': 250,
                   'max_depth': 13,
                   'learning_rate': 0.05,
                   'subsample': 0.7,
                   'colsample_bytree': 0.8,
                   'colsample_bynode': 0.7 ,
                   'path_smooth': 0.2,
                   'num_leaves': 70,
                   'min_child_samples': 40,
                   # 'data_sample_strategy': 'goss'
                   }

save_model_to_dir = '../../Model_Run/Model_trained'
makedirs([save_model_to_dir])

load_model = False
save_model = True
model_name = f'trial_effective_precip_{model_version}.joblib'

lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict, model='lgbm', n_jobs=-1,
                               load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                               model_save_name=model_name, tune_hyperparameters=False, repeated_Kfold=False,
                               n_folds=5, n_iter=10, n_repeats=5)

print(lgbm_reg_trained)

########################################################################################################################
print('##################################')

# checking train accuracy
train_test_pred_dir = '../../Model_Run/Model_csv/prediction'
makedirs([train_test_pred_dir])

y_pred_train = lgbm_reg_trained.predict(x_train)
train_rmse = calculate_rmse(Y_pred=y_pred_train, Y_obsv=y_train)
train_r2 = calculate_r2(Y_pred=y_pred_train, Y_obsv=y_train)

print(f'Train RMSE = {round(train_rmse, 4)} for random split ')
print(f'Train R2 = {round(train_r2, 4)} for random split')
print('\n')

# saving train prediction
train_obsv_predict_df = pd.DataFrame({'train_observed': y_train.values.ravel(),
                                       'train_predicted': y_pred_train})
train_obsv_predict_df.to_csv(os.path.join(train_test_pred_dir, f'train_obsv_pred_{model_version}.csv'), index=False)

# checking test accuracy
y_pred_test = lgbm_reg_trained.predict(x_test)
test_rmse = calculate_rmse(Y_pred=y_pred_test, Y_obsv=y_test)
test_r2 = calculate_r2(Y_pred=y_pred_test, Y_obsv=y_test)

print(f'Test RMSE = {round(test_rmse, 4)} for random split ')
print(f'Test R2 = {round(test_r2, 4)} for random split')

# saving test prediction
test_obsv_predict_df = pd.DataFrame({'test_observed': y_test.values.ravel(),
                                      'test_predicted': y_pred_test})
test_obsv_predict_df.to_csv(os.path.join(train_test_pred_dir, f'test_obsv_pred_{model_version}.csv'), index=False)


# plotting results
plot_dir = '../../Model_Run/Plots'

scatter_plot_name = f'train_performance_scatter_{model_version}.tif'

scatter_plot_of_same_vars(Y_pred=y_pred_train, Y_obsv=y_train.to_numpy(),
                          alpha=0.03, color_format='o', marker_size=1,
                          x_label='Eff. Precip. Observed (mm)', y_label='Eff. Precip. Predicted (mm)',
                          title='For random train-test split',
                          plot_name=scatter_plot_name, savedir=plot_dir)

scatter_plot_name = f'test_performance_scatter_{model_version}.tif'
scatter_plot_of_same_vars(Y_pred=y_pred_test, Y_obsv=y_test.to_numpy(),
                          alpha=0.03, color_format='o', marker_size=1,
                          x_label='Eff. Precip. Observed (mm)', y_label='Eff. Precip. Predicted (mm)',
                          title='For random train-test split',
                          plot_name=scatter_plot_name, savedir=plot_dir)

density_plot_name = f'train_performance_density_{model_version}.tif'
density_grid_plot_of_same_vars(Y_pred=y_pred_train, Y_obsv=y_train.to_numpy().ravel(),
                               x_label='Eff. Precip. Observed (mm)', y_label='Eff. Precip. Predicted (mm)',
                               plot_name=density_plot_name, savedir=plot_dir,
                               bins=80, title='For random train-test split',
                               axis_lim=None)

density_plot_name = f'test_performance_density_{model_version}.tif'
density_grid_plot_of_same_vars(Y_pred=y_pred_test, Y_obsv=y_test.to_numpy().ravel(),
                               x_label='Eff. Precip. Observed (mm)', y_label='Eff. Precip. Predicted (mm)',
                               plot_name=density_plot_name, savedir=plot_dir,
                               bins=80, title='For random train-test split',
                               axis_lim=None)

print('##################################')
########################################################################################################################

# ########################################################################################################################
# # Yearly model performance check
# skip_yearly_train_test_split = False
#
# year_available_for_train_test = [2016, 2017, 2018, 2019, 2020]
#
# for yr in year_available_for_train_test:
#     train_years = [i for i in year_available_for_train_test if i != yr]
#     test_year = [yr]
#
#     yearly_output_dir = os.path.join('../../Data_main/Model_csv/yearly_train_test_splits', f'test_year_{yr}')
#     x_train, x_test, y_train, y_test = \
#         split_train_val_test_set_by_year(input_csv=train_test_parquet_path,
#                                          pred_attr='Rainfed_cropET', exclude_columns=exclude_columns,
#                                          years_in_train=train_years, year_in_test=test_year,
#                                          output_dir=yearly_output_dir, verbose=False,
#                                          skip_processing=skip_yearly_train_test_split)
#
#     lgbm_param_dict = {'n_estimators': 250,
#                        'max_depth': 13,
#                        'learning_rate': 0.05,
#                        'subsample': 0.7,
#                        'colsample_bytree': 0.8,
#                        'colsample_bynode': 0.7,
#                        'path_smooth': 0.2,
#                        'num_leaves': 70,
#                        'min_child_samples': 40,
#                        # 'data_sample_strategy': 'goss'
#                        }
#
#     save_model_to_dir = '../../Model_Run/Model_trained'
#     load_model = False
#     save_model = True
#
#     lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict, model='lgbm', n_jobs=-1,
#                                    load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
#                                    model_save_name=f'trial_effective_precip_{yr}.joblib',
#                                    tune_hyperparameters=False, repeated_Kfold=False, n_folds=5, n_iter=10, n_repeats=5)
#
#     print('**************************')
#     # Train and Test accuracy for each year
#     y_pred_train = lgbm_reg_trained.predict(x_train)
#     train_rmse = calculate_rmse(Y_pred=y_pred_train, Y_obsv=y_train)
#     train_r2 = calculate_r2(Y_pred=y_pred_train, Y_obsv=y_train)
#
#     print(f'For year = {yr} as test set, Train RMSE = {round(train_rmse, 4)}')
#     print(f'For year = {yr} as test set, Train R2 = {round(train_r2, 4)}')
#     print('\n')
#
#     y_pred_test = lgbm_reg_trained.predict(x_test)
#     test_rmse = calculate_rmse(Y_pred=y_pred_test, Y_obsv=y_test)
#     test_r2 = calculate_r2(Y_pred=y_pred_test, Y_obsv=y_test)
#
#     print(f'For year = {yr} as test set, Test RMSE = {round(test_rmse, 4)}')
#     print(f'For year = {yr} as test set, Test R2 = {round(test_r2, 4)}')
#     print('**************************')
#
#     scatter_plot_name = f'scatter_test_year_{yr}.tif'
#     scatter_plot_dir = '../../Model_Run/Plots'
#
#     scatter_plot_of_same_vars(Y_pred=y_pred_test, Y_obsv=y_test.to_numpy(),
#                  x_label='Eff. Precip. Observed (mm)', y_label='Eff. Precip. Predicted (mm)',
#                  title=f'For test_year_{yr}',
#                  plot_name=scatter_plot_name, savedir=scatter_plot_dir)
#
# ########################################################################################################################

def create_pdplots(trained_model, x_train, features_to_include, output_dir, plot_name):
    """

    :param trained_model:
    :param x_train:
    :param features_to_include:
    :param output_dir:
    :param plot_name:
    :return:
    """
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
        'GRIDMET_wind_vel': 'GRIDMET Wind Velocity (m/s)', 'GRIDMET_short_rad': 'GRIDMET Downward Shortwave Radiation W/m^2',
        'DAYMET_sun_hr': 'DAYMET Daylight Duration (hr)', 'Bulk_density': 'Bulk Density (kg/m^3)',
        'Clay_content': 'Clay Content (%)', 'Field_capacity': 'Field Capacity (%)', 'Sand_content': 'Sand Content (%)',
        'Slope': 'Slope (%)', 'Latitude': f'Latitude ({deg_unit})', 'Longitude': f'Longitude ({deg_unit})'
    }

    # renaming columns
    x_train = x_train.rename(columns=feature_dict)

    # plotting
    if features_to_include == 'All':  # to plot PDP for all attributes
        features_to_include = list(x_train.columns)

    pdisp = PDisp.from_estimator(trained_model, x_train, features=features_to_include,
                                 percentiles=(0.05, 0.95), subsample=0.8, grid_resolution=20,
                                 n_jobs=-1, random_state=0)

    # replacing Y axis labels
    for row_idx in range(0, pdisp.axes_.shape[0]):
        pdisp.axes_[row_idx][0].set_ylabel('Effective Precipitation')

    fig = plt.gcf()
    fig.set_size_inches(30, 25)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(output_dir, plot_name), dpi=300, bbox_inches='tight')

create_pdplots(trained_model=lgbm_reg_trained, x_train=x_train,
               features_to_include='All', output_dir=plot_dir, plot_name=f'pdp_{model_version}.tif')
