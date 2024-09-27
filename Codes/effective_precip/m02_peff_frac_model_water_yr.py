import os
import sys
import pandas as pd

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_r2, calculate_rmse, calculate_mae
from Codes.utils.plots import scatter_plot_of_same_vars, density_grid_plot_of_same_vars
from Codes.utils.ml_ops import create_train_test_annual_dataframe, split_train_val_test_set, train_model, \
    create_pdplots, plot_permutation_importance
from Codes.effective_precip.m00_eff_precip_utils import create_annual_dataframes_for_peff_frac_prediction, \
    create_nan_pos_dict_for_annual_irrigated_cropET, create_annual_peff_fraction_rasters, \
    collect_Peff_predictions_in_dataframe

# model resolution and reference raster/shapefile
no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

# predictor data paths
yearly_data_path_dict = {
    'Peff_frac': '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/rainfed_cropET_water_year_fraction',
    'PRISM_Precip': '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year',
    'PRISM_Tmax': '../../Data_main/Raster_data/PRISM_Tmax/WestUS_water_year',
    'PRISM_Tmin': '../../Data_main/Raster_data/PRISM_Tmin/WestUS_water_year',
    'GRIDMET_Precip': '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year',
    'GRIDMET_RET': '../../Data_main/Raster_data/GRIDMET_RET/WestUS_water_year',
    'GRIDMET_vap_pres_def': '../../Data_main/Raster_data/GRIDMET_vap_pres_def/WestUS_water_year',
    'GRIDMET_max_RH': '../../Data_main/Raster_data/GRIDMET_max_RH/WestUS_water_year',
    'GRIDMET_min_RH': '../../Data_main/Raster_data/GRIDMET_min_RH/WestUS_water_year',
    'GRIDMET_wind_vel': '../../Data_main/Raster_data/GRIDMET_wind_vel/WestUS_water_year',
    'GRIDMET_short_rad': '../../Data_main/Raster_data/GRIDMET_short_rad/WestUS_water_year',
    'DAYMET_sun_hr': '../../Data_main/Raster_data/DAYMET_sun_hr/WestUS_water_year',
    'TERRACLIMATE_SR': '../../Data_main/Raster_data/TERRACLIMATE_SR/WestUS_water_year'}

static_data_path_dict = {
    'Bulk_density': '../../Data_main/Raster_data/Bulk_density/WestUS',
    'Clay_content': '../../Data_main/Raster_data/Clay_content/WestUS',
    'Field_capacity': '../../Data_main/Raster_data/Field_capacity/WestUS',
    'Sand_content': '../../Data_main/Raster_data/Sand_content/WestUS',
    'DEM': '../../Data_main/Raster_data/DEM/WestUS',
    'Slope': '../../Data_main/Raster_data/Slope/WestUS',
    'AWC': '../../Data_main/Raster_data/Available_water_capacity/WestUS',
    'Latitude': '../../Data_main/Raster_data/Latitude/WestUS',
    'Longitude': '../../Data_main/Raster_data/Longitude/WestUS'}

# datasets to include in the dataframe (not all will go into the final model)
datasets_to_include = ['Peff_frac',
                       'PRISM_Tmax', 'PRISM_Tmin', 'GRIDMET_Precip', 'GRIDMET_RET',
                       'GRIDMET_vap_pres_def', 'GRIDMET_max_RH', 'GRIDMET_min_RH',
                       'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                       'TERRACLIMATE_SR', 'Bulk_density', 'Clay_content',
                       'Field_capacity', 'Sand_content', 'AWC', 'DEM',
                       'Slope', 'Latitude', 'Longitude']

# exclude columns during training
exclude_columns_in_training = ['PRISM_Tmin',
                               'GRIDMET_min_RH', 'GRIDMET_wind_vel',
                               'DAYMET_sun_hr', 'Field_capacity',
                               'Sand_content', 'Bulk_density', 'Clay_content',
                               'AWC', 'DEM', 'Latitude', 'Longitude']

# training time periods
train_test_years_list = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                         2020]  # training data starting from 2008 as rainfed cropET dataset starts from 2008

# datasets to include in annual  dataframe for Peff fraction prediction
datasets_to_include_annual_predictors = ['PRISM_Tmax', 'PRISM_Tmin', 'GRIDMET_Precip', 'GRIDMET_RET',
                                         'GRIDMET_vap_pres_def', 'GRIDMET_max_RH', 'GRIDMET_min_RH',
                                         'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                                         'TERRACLIMATE_SR', 'Bulk_density', 'Clay_content',
                                         'Field_capacity', 'Sand_content', 'AWC', 'DEM',
                                         'Slope', 'Latitude', 'Longitude']

# exclude columns during prediction (the prediction dataframes don't have 'year' column)
exclude_columns_in_prediction = exclude_columns_in_training

# prediction time periods
prediction_years = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                    2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

if __name__ == '__main__':
    model_version = 'v14'                                       ######

    skip_train_test_df_creation = True                         ######
    skip_train_test_split = False                               ######
    skip_tune_hyperparams = True                               ######
    load_model = False                                          ######
    save_model = True                                          ######
    skip_plot_pdp = False                                       ######
    skip_plot_perm_import = False                               ######
    skip_processing_annual_predictor_dataframe = False          ######
    skip_processing_nan_pos_irrig_cropET = False                ######
    skip_estimate_water_year_peff_frac_WestUS = False           ######
    skip_storing_peff_frac_pred_annual_csv = False              ######

    # ********************************* Dataframe creation and train-test split (westUS) ***********************************
    # # create dataframe
    print(f'Running model version {model_version}...')

    train_test_parquet_path = f'../../Eff_Precip_Model_Run/annual_model/Model_csv/train_test.parquet'
    makedirs([os.path.dirname(train_test_parquet_path)])

    compiled_parquet = create_train_test_annual_dataframe(years_list=train_test_years_list,
                                                          yearly_data_path_dict=yearly_data_path_dict,
                                                          static_data_path_dict=static_data_path_dict,
                                                          datasets_to_include=datasets_to_include,
                                                          output_parquet=train_test_parquet_path,
                                                          skip_processing=skip_train_test_df_creation,
                                                          n_partitions=5)

    # # train-test split
    output_dir = '../../Eff_Precip_Model_Run/annual_model/Model_csv'
    makedirs([output_dir])

    x_train, x_test, y_train, y_test = \
        split_train_val_test_set(input_csv=train_test_parquet_path, month_range=None,
                                 model_version=model_version,
                                 pred_attr='Peff_frac', exclude_columns=exclude_columns_in_training,
                                 output_dir=output_dir, test_perc=0.3, validation_perc=0,
                                 random_state=0, verbose=True,
                                 skip_processing=skip_train_test_split,
                                 remove_outlier=False, outlier_upper_val=None)

    # ******************************** Model training and performance evaluation (westUS) **********************************

    # # model training  (if hyperparameter tuning is on, the default parameter dictionary will be disregarded)
    print('########## Model training')
    lgbm_param_dict = {'boosting_type': 'gbdt',
                       'colsample_bynode': 0.85,
                       'colsample_bytree': 0.99,
                       'learning_rate': 0.099,
                       'max_depth': 14,
                       'min_child_samples': 20,
                       'n_estimators': 400,
                       'num_leaves': 70,
                       'path_smooth': 0.47,
                       'subsample': 0.75,
                       'data_sample_strategy': 'goss'}

    save_model_to_dir = '../../Eff_Precip_Model_Run/annual_model/Model_trained'
    makedirs([save_model_to_dir])

    model_name = f'effective_precip_frac_{model_version}.joblib'
    max_evals = 500  ######
    param_iteration_csv = '../../Eff_Precip_Model_Run/annual_model/hyperparam_tune/hyperparam_iteration.csv'

    lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict, n_jobs=-1,
                                   load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                                   model_save_name=model_name,
                                   skip_tune_hyperparameters=skip_tune_hyperparams,
                                   iteration_csv=param_iteration_csv, n_fold=10, max_evals=max_evals)
    print(lgbm_reg_trained)
    print('########## Model performance')

    # # model performance evaluation
    train_test_pred_dir = '../../Eff_Precip_Model_Run/annual_model/Model_csv/prediction_csv'
    makedirs([train_test_pred_dir])

    y_pred_train = lgbm_reg_trained.predict(x_train)
    train_rmse = calculate_rmse(Y_pred=y_pred_train, Y_obsv=y_train)
    train_r2 = calculate_r2(Y_pred=y_pred_train, Y_obsv=y_train)
    train_mae = calculate_mae(Y_pred=y_pred_train, Y_obsv=y_train)

    print(f'Train RMSE = {round(train_rmse, 4)} for random split')
    print(f'Train MAE = {round(train_mae, 4)} for random split')
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
    test_mae = calculate_mae(Y_pred=y_pred_test, Y_obsv=y_test)

    print(f'Test RMSE = {round(test_rmse, 4)} for random split')
    print(f'Test MAE = {round(test_mae, 4)} for random split ')
    print(f'Test R2 = {round(test_r2, 4)} for random split')

    # saving test prediction
    test_obsv_predict_df = pd.DataFrame({'test_observed': y_test.values.ravel(),
                                         'test_predicted': y_pred_test})
    test_obsv_predict_df.to_csv(os.path.join(train_test_pred_dir, f'test_obsv_pred_{model_version}.csv'), index=False)

    # plotting results
    plot_dir = '../../Eff_Precip_Model_Run/annual_model/Plots'

    # scatter plots
    scatter_plot_name = f'train_performance_scatter_{model_version}.png'

    scatter_plot_of_same_vars(Y_pred=train_obsv_predict_df['train_predicted'],
                              Y_obsv=train_obsv_predict_df['train_observed'],
                              alpha=0.1, color_format='o', marker_size=1,
                              x_label='Reference effective precipitation \n fraction',
                              y_label='Predicted effective precipitation \n fraction',
                              title='For random train-test split',
                              plot_name=scatter_plot_name,
                              savedir=plot_dir, tick_interval=0.25)

    scatter_plot_name = f'test_performance_scatter_{model_version}.png'
    scatter_plot_of_same_vars(Y_pred=test_obsv_predict_df['test_predicted'],
                              Y_obsv=test_obsv_predict_df['test_observed'],
                              alpha=0.1, color_format='o', marker_size=1,
                              x_label='Reference effective precipitation \n fraction',
                              y_label='Predicted effective precipitation \n fraction',
                              title='For random train-test split',
                              plot_name=scatter_plot_name,
                              savedir=plot_dir, tick_interval=0.25)

    # density plots
    train_obsv_predict_df_copy = train_obsv_predict_df.copy()  # copied to add zero in the beginning for density grid plot
    test_obsv_predict_df_copy = test_obsv_predict_df.copy()  # copied to add zero in the beginning for density grid plot

    train_obsv_predict_df_copy.iloc[0, :] = 0  # added zero in 1st row for density grid plot
    test_obsv_predict_df_copy.iloc[0, :] = 0  # added zero in 1st row for density grid plot

    density_plot_name = f'train_performance_density_{model_version}.png'
    density_grid_plot_of_same_vars(Y_pred=train_obsv_predict_df_copy['train_predicted'],
                                   Y_obsv=train_obsv_predict_df_copy['train_observed'],
                                   x_label='Reference effective precipitation \n fraction',
                                   y_label='Predicted effective precipitation \n fraction',
                                   plot_name=density_plot_name, savedir=plot_dir,
                                   bins=100, tick_interval=0.25)

    density_plot_name = f'test_performance_density_{model_version}.png'
    density_grid_plot_of_same_vars(Y_pred=test_obsv_predict_df_copy['test_predicted'],
                                   Y_obsv=test_obsv_predict_df_copy['test_observed'],
                                   x_label='Reference effective precipitation \n fraction',
                                   y_label='Predicted effective precipitation \n fraction',
                                   plot_name=density_plot_name, savedir=plot_dir,
                                   bins=100, tick_interval=0.25)

    # partial dependence plots (pdp)
    features_in_pdp_plot = ['GRIDMET_RET', 'GRIDMET_Precip', 'GRIDMET_vap_pres_def',
                            'PRISM_Tmax', 'GRIDMET_max_RH', 'GRIDMET_short_rad',
                            'Slope', 'TERRACLIMATE_SR']

    create_pdplots(trained_model=lgbm_reg_trained, x_train=x_train,
                   features_to_include=features_in_pdp_plot, output_dir=plot_dir,
                   plot_name=f'pdp_{model_version}.png', ylabel='Effective precipitation \n fraction',
                   skip_processing=skip_plot_pdp)

    # permutation importance plot
    plot_permutation_importance(trained_model=lgbm_reg_trained, x_test=x_test, y_test=y_test,
                                exclude_columns=None, output_dir=plot_dir, plot_name=f'perm_import_{model_version}',
                                skip_processing=skip_plot_perm_import)

    print('##################################')

    # ************************ Generating monthly effective precip estimates for 17 states (westUS) ************************
    print('**********************************')

    # # Creating monthly predictor dataframe for model prediction
    annual_predictor_csv_dir = '../../Eff_Precip_Model_Run/annual_model/Model_csv/annual_predictors'
    create_annual_dataframes_for_peff_frac_prediction(years_list=prediction_years,
                                                      yearly_data_path_dict=yearly_data_path_dict,
                                                      static_data_path_dict=static_data_path_dict,
                                                      datasets_to_include=datasets_to_include_annual_predictors,
                                                      output_dir=annual_predictor_csv_dir,
                                                      skip_processing=skip_processing_annual_predictor_dataframe)

    # # Creating nan position dict for irrigated cropET (westUS)
    irrigated_cropET_water_year_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_water_year'
    output_dir_nan_pos = '../../Eff_Precip_Model_Run/annual_model/Model_csv/nan_pos_irrigated_cropET'

    create_nan_pos_dict_for_annual_irrigated_cropET(irrigated_cropET_dir=irrigated_cropET_water_year_dir,
                                                    output_dir=output_dir_nan_pos,
                                                    skip_processing=skip_processing_nan_pos_irrig_cropET)

    # # Generating monthly Peff predictions for 17 states
    peff_fraction_water_year_output_dir = f'../../Data_main/Raster_data/Effective_precip_fraction_WestUS/{model_version}_water_year_frac'

    create_annual_peff_fraction_rasters(trained_model=lgbm_reg_trained, input_csv_dir=annual_predictor_csv_dir,
                                        exclude_columns=exclude_columns_in_prediction,
                                        irrig_cropET_nan_pos_dir=output_dir_nan_pos, ref_raster=WestUS_raster,
                                        prediction_name_keyword='peff_frac',
                                        output_dir=peff_fraction_water_year_output_dir,
                                        skip_processing=skip_estimate_water_year_peff_frac_WestUS)

    # # # Storing annual predictions of Peff for all years_list in a dataframe
    output_csv = os.path.join(
        f'../../Data_main/Raster_data/Effective_precip_fraction_WestUS/{model_version}_annual.csv')
    collect_Peff_predictions_in_dataframe(input_peff_dir=peff_fraction_water_year_output_dir,
                                          output_csv=output_csv, skip_processing=skip_storing_peff_frac_pred_annual_csv)

    print('**********************************')
