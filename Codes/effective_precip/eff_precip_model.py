import os
import sys
import pandas as pd

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.stats_ops import calculate_r2, calculate_rmse
from Codes.utils.plots import scatter_plot_of_same_vars, density_grid_plot_of_same_vars
from Codes.utils.ml_ops import create_train_test_dataframe, split_train_val_test_set, \
    train_model, create_pdplots

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


monthly_data_path_dict = {'Irrigated_cropET': '../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly',
                          'Rainfed_cropET': '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly',
                          'Effective_precip_train': '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/final_filtered_cropET_for_training',
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
                         'Irrigated_Frac': '../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
                         'Rainfed_Frac': '../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac'}

static_data_path_dict = {'Bulk_density': '../../Data_main/Raster_data/Bulk_density/WestUS',
                         'Clay_content': '../../Data_main/Raster_data/Clay_content/WestUS',
                         'Field_capacity': '../../Data_main/Raster_data/Field_capacity/WestUS',
                         'Organic_carbon_content': '../../Data_main/Raster_data/Organic_carbon_content/WestUS',
                         'Sand_content': '../../Data_main/Raster_data/Sand_content/WestUS',
                         'DEM': '../../Data_main/Raster_data/DEM/WestUS',
                         'Slope': '../../Data_main/Raster_data/Slope',
                         'Latitude': '../../Data_main/Raster_data/Latitude',
                         'Longitude': '../../Data_main/Raster_data/Longitude'}

train_test_years_list = [2016, 2017, 2018, 2019, 2020]
total_month_range = (1, 12)  # considering March-November as growing season

datasets_to_include = ['Effective_precip_train', 'Rainfed_Frac',
                       'MODIS_Day_LST', 'MODIS_LAI', 'MODIS_NDVI', 'MODIS_NDWI',
                       'PRISM_Precip', 'PRISM_Tmax', 'PRISM_Tmin', 'Ssebop_ETa',
                       'GRIDMET_Precip', 'GRIDMET_RET', 'GRIDMET_vap_pres_def', 'GRIDMET_max_RH',
                       'GRIDMET_min_RH', 'GRIDMET_wind_vel', 'GRIDMET_short_rad', 'DAYMET_sun_hr',
                       'Bulk_density', 'Clay_content', 'Field_capacity', 'Sand_content',
                       'DEM', 'Slope', 'Latitude', 'Longitude']

skip_train_test_df_creation = True  # #
train_test_parquet_path = '../../Model_Run/Model_csv/train_test.parquet'

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
print('Available columns in the compiled training dataframe:')
print(compiled_training_df.columns)


output_dir = '../../Model_Run/Model_csv'
makedirs([output_dir])

model_version = 'v3'  # #
train_test_month_range = (4, 10)
skip_train_test_split = False  # #

# ********* 'Rainfed_Frac' was renamed to 'Cropland_Frac' *********
exclude_columns = ['year', 'month',
                   'Ssebop_ETa', 'GRIDMET_Precip', 'GRIDMET_max_RH', 'GRIDMET_min_RH',
                   'GRIDMET_wind_vel', 'MODIS_NDWI', 'PRISM_Tmax', 'Cropland_Frac',
                   'Latitude', 'Bulk_density', 'Field_capacity', 'Clay_content', 'Slope']


x_train, x_test, y_train, y_test = \
    split_train_val_test_set(input_csv=train_test_parquet_path, month_range=train_test_month_range,
                             model_version=model_version,
                             pred_attr='Effective_precip_train', exclude_columns=exclude_columns,
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
train_test_pred_dir = '../../Model_Run/Model_csv/prediction_csv'
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
#                                          general_output_dir=yearly_output_dir, verbose=False,
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

create_pdplots(trained_model=lgbm_reg_trained, x_train=x_train,
               features_to_include='All', output_dir=plot_dir, plot_name=f'pdp_{model_version}.tif')
