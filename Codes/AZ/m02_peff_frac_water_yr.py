import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_r2, calculate_rmse, calculate_mae
from Codes.utils.ml_ops import split_train_val_test_set, train_model
from Codes.effective_precip.m00_eff_precip_utils import create_annual_dataframes_for_peff_frac_prediction, \
    create_nan_pos_dict_for_annual_irrigated_cropET, create_annual_peff_fraction_rasters

# model resolution and reference raster/shapefile
no_data_value = -9999
model_res = 2000  # in m
AZ_shape = '../../Data_main/AZ_files/ref_files/AZ.shp'
AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'
GEE_merging_refraster_large_grids = '../../Data_main/AZ_files/ref_files/AZ_gee_merge_ref_raster.tif'

# predictor data paths
yearly_data_path_dict = {
    'PRISM_Tmax': '../../Data_main/AZ_files/rasters/PRISM_Tmax/WestUS_water_year',
    'GRIDMET_Precip': '../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_water_year/sum',
    'GRIDMET_short_rad': '../../Data_main/AZ_files/rasters/GRIDMET_short_rad/WestUS_water_year',
    'Precipitation_intensity': '../../Data_main/AZ_files/rasters/Precipitation_intensity',
    'Dryness_index': '../../Data_main/AZ_files/rasters/Dryness_index',
}

static_data_path_dict = {
    'PET_P_corr': '../../Data_main/AZ_files/rasters/P_PET_correlation',
    'Field_capacity': '../../Data_main/AZ_files/rasters/Field_capacity/WestUS',
    'Slope': '../../Data_main/AZ_files/rasters/Slope/WestUS',
}

# datasets to include in the dataframe (not all will go into the final model)
datasets_to_include = ['PRISM_Tmax', 'GRIDMET_Precip', 'GRIDMET_short_rad',
                       'Precipitation_intensity', 'Dryness_index',
                       'PET_P_corr', 'Field_capacity', 'Slope']

# exclude columns during training
exclude_columns_in_training = ['Field_capacity']


# datasets to include in annual  dataframe for Peff fraction prediction
datasets_to_include_annual_predictors = ['PRISM_Tmax', 'GRIDMET_Precip',
                                         'GRIDMET_short_rad', 'Precipitation_intensity',
                                         'Dryness_index', 'PET_P_corr',
                                         'Field_capacity', 'Slope']

# exclude columns during prediction (the prediction dataframes don't have 'year' column)
exclude_columns_in_prediction = exclude_columns_in_training

# prediction time periods
prediction_years = list(range(1986, 2021))       # couldn't do 1985 as ET data (needed for nan-pos) for 1984 not available

if __name__ == '__main__':
    model_version = 'v20'                                      ######

    skip_train_test_split = True                               ######
    load_model = True                                          ######
    save_model = False                                          ######
    skip_processing_annual_predictor_dataframe = False          ######
    skip_processing_nan_pos_irrig_cropET = False                ######
    skip_estimate_water_year_peff_frac_WestUS = False           ######

    # ******************************* Dataframe creation and train-test split (westUS) *********************************
    # # create dataframe
    print(f'Running model version {model_version}...')

    # # train-test split
    train_test_parquet_path = f'../../Eff_Precip_Model_Run/annual_model/Model_csv/train_test.parquet'
    output_dir = '../../Eff_Precip_Model_Run/annual_model/Model_csv'

    x_train, x_test, y_train, y_test = \
        split_train_val_test_set(input_csv=train_test_parquet_path, month_range=None,
                                 model_version=model_version,
                                 pred_attr='Peff_frac', exclude_columns=exclude_columns_in_training,
                                 output_dir=output_dir, test_perc=0.3, validation_perc=0,
                                 random_state=0, verbose=True,
                                 skip_processing=skip_train_test_split,
                                 remove_outlier=False, outlier_upper_val=None)

    # ****************************** Model training and performance evaluation (westUS) ********************************

    # # model training  (if hyperparameter tuning is on, the default parameter dictionary will be disregarded)
    print('########## Model training')
    lgbm_param_dict = {'boosting_type': 'gbdt',
                       'colsample_bynode': 0.78,
                       'colsample_bytree': 0.95,
                       'learning_rate': 0.099,
                       'max_depth': 13,
                       'min_child_samples': 20,
                       'n_estimators': 400,
                       'num_leaves': 70,
                       'path_smooth': 0.406,
                       'subsample': 0.77,
                       'data_sample_strategy': 'goss'}

    save_model_to_dir = '../../Eff_Precip_Model_Run/annual_model/Model_trained'
    makedirs([save_model_to_dir])

    model_name = f'effective_precip_frac_{model_version}.joblib'

    lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict, n_jobs=-1,
                                   load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                                   model_save_name=model_name,
                                   skip_tune_hyperparameters=True,
                                   iteration_csv=None, n_fold=10, max_evals=1)
    print(lgbm_reg_trained)
    print('########## Model performance')

    # # model performance evaluation
    y_pred_train = lgbm_reg_trained.predict(x_train)
    train_rmse = calculate_rmse(Y_pred=y_pred_train, Y_obsv=y_train)
    train_r2 = calculate_r2(Y_pred=y_pred_train, Y_obsv=y_train)
    train_mae = calculate_mae(Y_pred=y_pred_train, Y_obsv=y_train)

    print(f'Train RMSE = {round(train_rmse, 4)} for random split')
    print(f'Train MAE = {round(train_mae, 4)} for random split')
    print(f'Train R2 = {round(train_r2, 4)} for random split')
    print('\n')

    # checking test accuracy
    y_pred_test = lgbm_reg_trained.predict(x_test)
    test_rmse = calculate_rmse(Y_pred=y_pred_test, Y_obsv=y_test)
    test_r2 = calculate_r2(Y_pred=y_pred_test, Y_obsv=y_test)
    test_mae = calculate_mae(Y_pred=y_pred_test, Y_obsv=y_test)

    print(f'Test RMSE = {round(test_rmse, 4)} for random split')
    print(f'Test MAE = {round(test_mae, 4)} for random split ')
    print(f'Test R2 = {round(test_r2, 4)} for random split')

    # ****************** Generating annual effective precip fraction estimates for 17 states (westUS) ******************
    print('**********************************')

    # # Creating water year predictor dataframe for model prediction
    annual_predictor_csv_dir = '../../Eff_Precip_Model_Run/AZ_model/water_yr/Model_csv/annual_predictors'
    create_annual_dataframes_for_peff_frac_prediction(years_list=prediction_years,
                                                      yearly_data_path_dict=yearly_data_path_dict,
                                                      static_data_path_dict=static_data_path_dict,
                                                      datasets_to_include=datasets_to_include_annual_predictors,
                                                      output_dir=annual_predictor_csv_dir,
                                                      skip_processing=skip_processing_annual_predictor_dataframe)

    # # Creating nan position dict for irrigated cropET (westUS)
    irrigated_cropET_water_year_dir = '../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_water_year'
    output_dir_nan_pos = '../../Eff_Precip_Model_Run/AZ_model/water_yr/Model_csv/nan_pos_irrigated_cropET'

    create_nan_pos_dict_for_annual_irrigated_cropET(irrigated_cropET_dir=irrigated_cropET_water_year_dir,
                                                    output_dir=output_dir_nan_pos,
                                                    skip_processing=skip_processing_nan_pos_irrig_cropET)

    # # Generating water year Peff fraction predictions
    peff_fraction_water_year_output_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_fraction_WestUS/{model_version}_water_year_frac'

    create_annual_peff_fraction_rasters(trained_model=lgbm_reg_trained, input_csv_dir=annual_predictor_csv_dir,
                                        exclude_columns=exclude_columns_in_prediction,
                                        irrig_cropET_nan_pos_dir=output_dir_nan_pos, ref_raster=AZ_raster,
                                        prediction_name_keyword='peff_frac',
                                        output_dir=peff_fraction_water_year_output_dir,
                                        lake_raster='../../Data_main/AZ_files/rasters/HydroLakes/Lakes_AZ.tif',
                                        skip_processing=skip_estimate_water_year_peff_frac_WestUS)
