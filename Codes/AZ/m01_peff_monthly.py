import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_r2, calculate_rmse, calculate_mae
from Codes.utils.ml_ops import split_train_val_test_set, train_model
from Codes.effective_precip.m00_eff_precip_utils import create_nan_pos_dict_for_monthly_irrigated_cropET, \
    create_monthly_effective_precip_rasters, sum_peff_water_year
from Codes.AZ.az_utils import create_monthly_dataframes_for_eff_precip_prediction

# model resolution and reference raster/shapefile
no_data_value = -9999
model_res = 2000  # in m
AZ_shape = '../../Data_main/AZ_files/ref_files/AZ.shp'
AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'
GEE_merging_refraster_large_grids = '../../Data_main/AZ_files/ref_files/AZ_gee_merge_ref_raster.tif'

# # predictor data paths
monthly_data_path_dict = {
    'GRIDMET_Precip': '../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_monthly',
    'GRIDMET_RET': '../../Data_main/AZ_files/rasters/GRIDMET_RET/WestUS_monthly',
    'GRIDMET_max_RH': '../../Data_main/AZ_files/rasters/GRIDMET_max_RH/WestUS_monthly',
    'GRIDMET_short_rad': '../../Data_main/AZ_files/rasters/GRIDMET_short_rad/WestUS_monthly',
    'GRIDMET_wind_vel': '../../Data_main/AZ_files/rasters/GRIDMET_wind_vel/WestUS_monthly',
    'DAYMET_sun_hr': '../../Data_main/AZ_files/rasters/DAYMET_sun_hr/WestUS_monthly'}

static_data_path_dict = {
    'Field_capacity': '../../Data_main/AZ_files/rasters/Field_capacity/WestUS',
    'Sand_content': '../../Data_main/AZ_files/rasters/Sand_content/WestUS',
    'Slope': '../../Data_main/AZ_files/rasters/Slope/WestUS',
    'AWC': '../../Data_main/AZ_files/rasters/Available_water_capacity/WestUS'}

# # exclude columns during training
exclude_columns_in_training = ['Slope']

# # training time periods
# train_test_years_list = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
months = (1, 12)  # the model will itself discard the month is places where growing season is less than 12 months
                  # (using the nan value set up)

# datasets to include in monthly dataframe for Peff prediction
datasets_to_include_month_predictors = ['GRIDMET_Precip', 'GRIDMET_RET', 'GRIDMET_max_RH',
                                        'GRIDMET_short_rad', 'DAYMET_sun_hr',
                                        'Field_capacity', 'Sand_content',
                                        'AWC', 'Slope']

# exclude columns during prediction (the prediction dataframes don't have 'year' column)
exclude_columns_in_prediction = ['Slope']


# prediction time periods
prediction_years = list(range(1985, 2021))  # 1985 to 2020

if __name__ == '__main__':
    model_version = 'v19'                                   ######

    skip_train_test_split = True                            ######
    load_model = True                                       ######
    save_model = False                                       ######
    skip_processing_monthly_predictor_dataframe = True      ######
    skip_processing_nan_pos_irrig_cropET = True            ######
    skip_estimate_monthly_eff_precip_AZ = False          ######
    skip_sum_peff_water_year = False                         ######

    # ********************************* Run-Load monthly Peff model ***********************************
    # # create dataframe
    print(f'Running monthly model version {model_version}...')

    # # train-test split
    train_test_parquet_path = f'../../Eff_Precip_Model_Run/monthly_model/Model_csv/train_test.parquet'
    output_dir = '../../Eff_Precip_Model_Run/monthly_model/Model_csv'
    makedirs([output_dir])

    x_train, x_test, y_train, y_test = \
        split_train_val_test_set(input_csv=train_test_parquet_path, month_range=months,
                                 model_version=model_version,
                                 pred_attr='Effective_precip_train', exclude_columns=exclude_columns_in_training,
                                 output_dir=output_dir, test_perc=0.3, validation_perc=0,
                                 random_state=0, verbose=True,
                                 skip_processing=skip_train_test_split,
                                 remove_outlier=False, outlier_upper_val=None)

    # ******************************** Model training and performance evaluation (westUS) **********************************

    # # model training  (if hyperparameter tuning is on, the default parameter dictionary will be disregarded)
    print('########## Model training')
    lgbm_param_dict = {'boosting_type': 'gbdt',
                       'colsample_bynode': 0.77,
                       'colsample_bytree': 0.96,
                       'learning_rate': 0.09,
                       'max_depth': 14,
                       'min_child_samples': 45,
                       'n_estimators': 400,
                       'num_leaves': 70,
                       'path_smooth': 0.26,
                       'subsample': 1,
                       'data_sample_strategy': 'goss'}

    save_model_to_dir = '../../Eff_Precip_Model_Run/monthly_model/Model_trained'
    makedirs([save_model_to_dir])

    model_name = f'effective_precip_{model_version}.joblib'

    lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict, n_jobs=-1,
                                   load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                                   model_save_name=model_name, skip_tune_hyperparameters=True,
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

    print('##################################')

    # ************************ Generating monthly effective precip estimates for 17 states (westUS) ************************
    print('**********************************')

    # # Creating monthly predictor dataframe for model prediction
    monthly_predictor_csv_dir = '../../Eff_Precip_Model_Run/AZ_model/monthly/Model_csv/monthly_predictors'
    create_monthly_dataframes_for_eff_precip_prediction(years_list=prediction_years,
                                                        month_range=months,
                                                        monthly_data_path_dict=monthly_data_path_dict,
                                                        yearly_data_path_dict=None,
                                                        static_data_path_dict=static_data_path_dict,
                                                        datasets_to_include=datasets_to_include_month_predictors,
                                                        output_dir=monthly_predictor_csv_dir,
                                                        skip_processing=skip_processing_monthly_predictor_dataframe)

    # # Creating nan position dict for irrigated cropET (westUS)
    irrigated_cropET_monthly_dir = '../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_monthly'
    output_dir_nan_pos = '../../Eff_Precip_Model_Run/AZ_model/monthly/Model_csv/nan_pos_irrigated_cropET'

    create_nan_pos_dict_for_monthly_irrigated_cropET(irrigated_cropET_dir=irrigated_cropET_monthly_dir,
                                                     output_dir=output_dir_nan_pos,
                                                     skip_processing=skip_processing_nan_pos_irrig_cropET)

    # # Generating monthly Peff predictions
    effective_precip_monthly_output_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{model_version}_monthly'

    create_monthly_effective_precip_rasters(trained_model=lgbm_reg_trained, input_csv_dir=monthly_predictor_csv_dir,
                                            exclude_columns=exclude_columns_in_prediction,
                                            irrig_cropET_nan_pos_dir=output_dir_nan_pos,
                                            prediction_name_keyword='effective_precip',
                                            output_dir=effective_precip_monthly_output_dir,
                                            ref_raster=AZ_raster,
                                            skip_processing=skip_estimate_monthly_eff_precip_AZ)


    # # summing monthly effective precipitation for water year
    water_yr_peff_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{model_version}_water_year'

    sum_peff_water_year(years_list=list(range(1985, 2021)),
                        monthly_peff_dir=effective_precip_monthly_output_dir,
                        output_peff_dir=water_yr_peff_dir, skip_processing=skip_sum_peff_water_year)

    print('**********************************')
