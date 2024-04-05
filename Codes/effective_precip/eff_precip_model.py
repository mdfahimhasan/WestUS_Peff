import os
import sys
import pandas as pd

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_r2, calculate_rmse
from Codes.utils.plots import scatter_plot_of_same_vars, density_grid_plot_of_same_vars
from Codes.utils.ml_ops import create_train_test_dataframe, split_train_val_test_set, train_model, \
    create_pdplots, plot_permutation_importance
from Codes.effective_precip.eff_precip_utils import filter_effective_precip_training_data, \
    create_monthly_dataframes_for_eff_precip_prediction, create_nan_pos_dict_for_irrigated_cropET, \
    create_monthly_effective_precip_rasters, sum_monthly_effective_precip_rasters

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

# ****************************** Filtering training data for effective precip (westUS) *********************************

# # filtering training data
training_zone_shapefile = '../../Data_main/shapefiles/Training_zones/effective_precip_training_zone.shp'
output_dir = '../../Data_main/Raster_data/Rainfed_cropET_filtered_training'
skip_effective_precip_training_data_filtering = False  ######

filter_effective_precip_training_data(training_zone_shp=training_zone_shapefile,
                                      general_output_dir=output_dir,
                                      refraster=WestUS_raster,
                                      resolution=model_res,
                                      skip_processing=skip_effective_precip_training_data_filtering)

# ********************************* Dataframe creation and train-test split (westUS) ***********************************

# # create dataframe
monthly_data_path_dict = {
    'Effective_precip_train': '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/final_filtered_cropET_for_training',
    'PRISM_Precip': '../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly',
    'PRISM_Tmax': '../../Data_main/Raster_data/PRISM_Tmax/WestUS_monthly',
    'PRISM_Tmin': '../../Data_main/Raster_data/PRISM_Tmin/WestUS_monthly',
    'GRIDMET_Precip': '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_monthly',
    'GRIDMET_RET': '../../Data_main/Raster_data/GRIDMET_RET/WestUS_monthly',
    'GRIDMET_vap_pres_def': '../../Data_main/Raster_data/GRIDMET_vap_pres_def/WestUS_monthly',
    'GRIDMET_max_RH': '../../Data_main/Raster_data/GRIDMET_max_RH/WestUS_monthly',
    'GRIDMET_min_RH': '../../Data_main/Raster_data/GRIDMET_min_RH/WestUS_monthly',
    'GRIDMET_wind_vel': '../../Data_main/Raster_data/GRIDMET_wind_vel/WestUS_monthly',
    'GRIDMET_short_rad': '../../Data_main/Raster_data/GRIDMET_short_rad/WestUS_monthly',
    'DAYMET_sun_hr': '../../Data_main/Raster_data/DAYMET_sun_hr/WestUS_monthly'}

yearly_data_path_dict = None

static_data_path_dict = {'Bulk_density': '../../Data_main/Raster_data/Bulk_density/WestUS',
                         'Clay_content': '../../Data_main/Raster_data/Clay_content/WestUS',
                         'Field_capacity': '../../Data_main/Raster_data/Field_capacity/WestUS',
                         'Sand_content': '../../Data_main/Raster_data/Sand_content/WestUS',
                         'DEM': '../../Data_main/Raster_data/DEM/WestUS',
                         'Slope': '../../Data_main/Raster_data/Slope/WestUS',
                         'AWC': '../../Data_main/Raster_data/Available_water_capacity/WestUS',
                         'Latitude': '../../Data_main/Raster_data/Latitude/WestUS',
                         'Longitude': '../../Data_main/Raster_data/Longitude/WestUS'}

# # time periods
# training data starting from 2008 as rainfed cropET dataset starts from 2008
# considering growing season months for dataframe creation,
train_test_years_list = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
total_month_range = (4, 10)

model_version = 'v9'  ######
print(f'Running model version {model_version}...')

datasets_to_include = ['Effective_precip_train',
                       'PRISM_Precip', 'PRISM_Tmax', 'PRISM_Tmin',
                       'GRIDMET_Precip', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'GRIDMET_max_RH',
                       'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                       'Bulk_density', 'Clay_content', 'Field_capacity', 'Sand_content',
                       'AWC', 'DEM', 'Slope', 'Latitude', 'Longitude']

skip_train_test_df_creation = False  ######
train_test_parquet_path = f'../../Eff_Precip_Model_Run/Model_csv/train_test.parquet'

makedirs([os.path.dirname(train_test_parquet_path)])

compiled_parquet = create_train_test_dataframe(years_list=train_test_years_list,
                                               month_range=total_month_range,
                                               monthly_data_path_dict=monthly_data_path_dict,
                                               yearly_data_path_dict=yearly_data_path_dict,
                                               static_data_path_dict=static_data_path_dict,
                                               datasets_to_include=datasets_to_include,
                                               output_parquet=train_test_parquet_path,
                                               skip_processing=skip_train_test_df_creation,
                                               n_partitions=5)

compiled_training_df = pd.read_parquet(compiled_parquet)

# # train-test split
train_test_month_range = (4, 10)
output_dir = '../../Eff_Precip_Model_Run/Model_csv'
makedirs([output_dir])

exclude_columns = ['year', 'Latitude', 'Longitude',
                   'Bulk_density', 'Clay_content', 'Slope',
                   'PRISM_Tmax', 'PRISM_Tmin', 'PRISM_Precip',
                   'GRIDMET_wind_vel', 'GRIDMET_min_RH']

skip_train_test_split = False  ######

x_train, x_test, y_train, y_test = \
    split_train_val_test_set(input_csv=train_test_parquet_path, month_range=train_test_month_range,
                             model_version=model_version,
                             pred_attr='Effective_precip_train', exclude_columns=exclude_columns,
                             output_dir=output_dir, test_perc=0.3, validation_perc=0,
                             random_state=0, verbose=True,
                             skip_processing=skip_train_test_split,
                             remove_outlier=False, outlier_upper_val=None)

# ******************************** Model training and performance evaluation (westUS) **********************************

# # model training
print('########## Model training')
lgbm_param_dict = {'n_estimators': 250,
                   'max_depth': 13,
                   'learning_rate': 0.05,
                   'subsample': 0.7,
                   'colsample_bytree': 0.8,
                   'colsample_bynode': 0.7,
                   'path_smooth': 0.2,
                   'num_leaves': 70,
                   'min_child_samples': 40,
                   # 'data_sample_strategy': 'goss'
                   }

save_model_to_dir = '../../Eff_Precip_Model_Run/Model_trained'
makedirs([save_model_to_dir])

load_model = False
save_model = True
model_name = f'effective_precip_{model_version}.joblib'

lgbm_reg_trained = train_model(x_train=x_train, y_train=y_train, params_dict=lgbm_param_dict, model='lgbm', n_jobs=-1,
                               load_model=load_model, save_model=save_model, save_folder=save_model_to_dir,
                               model_save_name=model_name, tune_hyperparameters=False, repeated_Kfold=False,
                               n_folds=5, n_iter=10, n_repeats=5)

print(lgbm_reg_trained)

print('########## Model performacne')

# # model performance evaluation
train_test_pred_dir = '../../Eff_Precip_Model_Run/Model_csv/prediction_csv'
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
plot_dir = '../../Eff_Precip_Model_Run/Plots'

scatter_plot_name = f'train_performance_scatter_{model_version}.tif'

scatter_plot_of_same_vars(Y_pred=y_pred_train, Y_obsv=y_train.to_numpy(),
                          alpha=0.03, color_format='o', marker_size=1,
                          x_label='Eff. Precip. Observed (mm/month)', y_label='Eff. Precip. Predicted (mm/month)',
                          title='For random train-test split',
                          plot_name=scatter_plot_name, savedir=plot_dir)

scatter_plot_name = f'test_performance_scatter_{model_version}.tif'
scatter_plot_of_same_vars(Y_pred=y_pred_test, Y_obsv=y_test.to_numpy(),
                          alpha=0.03, color_format='o', marker_size=1,
                          x_label='Eff. Precip. Observed (mm/month)', y_label='Eff. Precip. Predicted (mm/month)',
                          title='For random train-test split',
                          plot_name=scatter_plot_name, savedir=plot_dir)

density_plot_name = f'train_performance_density_{model_version}.tif'
density_grid_plot_of_same_vars(Y_pred=y_pred_train, Y_obsv=y_train.to_numpy().ravel(),
                               x_label='Eff. Precip. Observed (mm/month)', y_label='Eff. Precip. Predicted (mm/month)',
                               plot_name=density_plot_name, savedir=plot_dir, bins=80)

density_plot_name = f'test_performance_density_{model_version}.tif'
density_grid_plot_of_same_vars(Y_pred=y_pred_test, Y_obsv=y_test.to_numpy().ravel(),
                               x_label='Eff. Precip. Observed (mm/month)', y_label='Eff. Precip. Predicted (mm/month)',
                               plot_name=density_plot_name, savedir=plot_dir, bins=80)

skip_plot_pdp = False  ######
deg_unit = r'$^\circ$'
features_in_pdp_plot = ['GRIDMET Precipitation (mm)', 'GRIDMET Reference ET (mm)',
                        'GRIDMET Mean Vapour Pressure Deficit (kpa)',
                        'GRIDMET Max Relative Humidity (%)',
                        'GRIDMET Downward Shortwave Radiation (W/m^2)',
                        'Field Capacity (%)', 'DEM', 'month']

create_pdplots(trained_model=lgbm_reg_trained, x_train=x_train,
               features_to_include=features_in_pdp_plot, output_dir=plot_dir,
               plot_name=f'pdp_{model_version}.tif', skip_processing=skip_plot_pdp)

skip_plot_perm_import = False  ######
plot_permutation_importance(trained_model=lgbm_reg_trained, x_test=x_test, y_test=y_test,
                            exclude_columns=None, output_dir=plot_dir, plot_name=f'perm_import_{model_version}',
                            skip_processing=skip_plot_perm_import)

print('##################################')

# ************************ Generating monthly effective precip estimates for 11 states (westUS) ************************
print('**********************************')

# # Creating monthly predictor dataframe for model prediction
datasets_to_include_month_predictors = ['PRISM_Precip', 'PRISM_Tmax', 'PRISM_Tmin',
                                        'GRIDMET_Precip', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'GRIDMET_max_RH',
                                        'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                                        'Bulk_density', 'Clay_content', 'Field_capacity', 'Sand_content',
                                        'AWC', 'DEM', 'Slope', 'Latitude', 'Longitude']

predictor_years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                   2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

monthly_predictor_csv_dir = '../../Eff_Precip_Model_Run/Model_csv/monthly_predictors'
skip_prcessing_monthly_predictor_dataframe = False
create_monthly_dataframes_for_eff_precip_prediction(years_list=predictor_years,
                                                    month_range=(4, 10),
                                                    monthly_data_path_dict=monthly_data_path_dict,
                                                    yearly_data_path_dict=yearly_data_path_dict,
                                                    static_data_path_dict=static_data_path_dict,
                                                    datasets_to_include=datasets_to_include_month_predictors,
                                                    output_dir=monthly_predictor_csv_dir,
                                                    skip_processing=skip_prcessing_monthly_predictor_dataframe)

# # Creating nan position dict for irrigated cropET (westUS)
irrigated_cropET_monthly_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly'
output_dir_nan_pos = '../../Eff_Precip_Model_Run/Model_csv/nan_pos_irrigated_cropET'
skip_processing_nan_pos_irrig_cropET = False

create_nan_pos_dict_for_irrigated_cropET(irrigated_cropET_dir=irrigated_cropET_monthly_dir,
                                         output_dir=output_dir_nan_pos,
                                         skip_processing=skip_processing_nan_pos_irrig_cropET)

# # Generating monthly predictions for 11 states
exclude_columns = ['Latitude', 'Longitude', 'Bulk_density', 'Clay_content',
                   'Slope', 'PRISM_Tmax', 'PRISM_Tmin', 'PRISM_Precip',
                   'GRIDMET_wind_vel', 'GRIDMET_min_RH']

effective_precip_monthly_output_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
skip_estimate_monthly_eff_precip_WestUS = False
create_monthly_effective_precip_rasters(trained_model=lgbm_reg_trained, input_csv_dir=monthly_predictor_csv_dir,
                                        exclude_columns=exclude_columns,
                                        irrig_cropET_nan_pos_dir=output_dir_nan_pos, ref_raster=WestUS_raster,
                                        prediction_name_keyword='effective_precip',
                                        output_dir=effective_precip_monthly_output_dir,
                                        skip_processing=skip_estimate_monthly_eff_precip_WestUS)

# # Summing monthly effective precipitation estimates for growing season
irrigated_cropET_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
grow_season_summed_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
skip_sum_effective_precip = False

sum_monthly_effective_precip_rasters(years_list=predictor_years,
                                     irrigated_cropET_dir=irrigated_cropET_dir,
                                     monthly_effective_precip_dir=effective_precip_monthly_output_dir,
                                     grow_season_effective_precip_output_dir=grow_season_summed_dir,
                                     skip_processing=skip_sum_effective_precip)

print('**********************************')
