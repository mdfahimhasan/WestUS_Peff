import pandas as pd

from ML_ops import create_dataframe_csv, create_train_val_test_data, train_nn_model, model_performance
from download_preprocess import download_all_datasets, run_all_preprocessing, compile_observed_pumping_data

# Data download
skip_download_gee_data = True
skip_download_ssebop_data = True

gee_data_list = ['MODIS_NDWI', 'GPM_PRECIP', 'MODIS_Day_LST', 'PRISM_PRECIP']
download_all_datasets(year_list=[2010, 2015], gee_data_list=gee_data_list,
                      skip_download_gee_data=skip_download_gee_data,
                      skip_download_ssebop_data=skip_download_ssebop_data)

# Compiling USGS county-scale GW data
include_years = '*201[0-5]*.xlsx'  # #
skip_compiling = True  # # Set to False to compile USGS GW use data

GW_usgs_csv = compile_observed_pumping_data\
    (data_dir='../Data_main/USGS_water_use_data', search_by=include_years,
     county_shape='../Data_main/shapefiles/Western_US/WestUS_county_projected.shp',
     output_csv='../Data_main/USGS_water_use_data/WestUS_county_gw_use.csv',
     skip_compiling=skip_compiling)


# Creating predictor csv/dataframe
exclude_data_from_df = ('MODIS_ET', 'MODIS_Terra_EVI', 'MODIS_Terra_NDVI', 'USDA_CDL')  # #
years = [2010, 2015]
skip_df_creation = True  # # Set to False to create predictor dataframe

predictor_WestUS = create_dataframe_csv(input_data_dir='../Data_main/Compiled_data',
                                        output_csv='../Data_main/Model_csv/WesternUS_data.csv',
                                        search_by='*.tif', years=years, drop_datasets=exclude_data_from_df,
                                        skip_dataframe_creation=skip_df_creation)

# Creating train, test, and validation dataset csv
data_frac, train_frac, val_frac, test_frac = 0.7, 0.7, 0.15, 0.15  # #
drop_predictor = ['USDA_cropland', 'USDA_developed']  # dropped attributes are are one-hot encoded and not necessary
train_val_test_exists = True  # # Set to False to create train, test, and validation dataset csv

train_csv, validation_csv, test_csv, train_obsv, validation_obsv, test_obsv = \
    create_train_val_test_data(predictor_csv=predictor_WestUS, observed_data_csv=GW_usgs_csv,
                               data_fraction=data_frac, train_fraction=train_frac, val_fraction=val_frac,
                               test_fraction=test_frac, output_dir='../Data_main/Model_csv',
                               drop_columns=drop_predictor, train_val_test_exists=train_val_test_exists)

# Model training
hidden_layers = [30, 20, 10, 5]  # #
activation = 'tanh'  # #
optimization = 'adam'  # #
epochs = 300  # #
learning_rate = 0.05  # #
device = 'cuda'  # #
rank = 0  # #
world_size = 1  # #
batch_size = int(len(pd.read_csv(train_csv))/20)  # batch_size optimization helps with CUDA memory error
verbose = True  # #
print_epoch = 25  # #
skip_train = False  # #
setup_ddp = True  # #

trained_model, rmse_loss, train_means, train_stds, obsv_mean, obsv_std = \
    train_nn_model(predictor_csv=train_csv, observed_csv=train_obsv, hidden_layers=hidden_layers,
                   activation=activation, optimization=optimization, epochs=epochs,
                   learning_rate=learning_rate, device=device, rank=rank, world_size=world_size,
                   batch_size=batch_size, num_workers=0, drop_columns=('fips', 'Year'),
                   verbose=True, epochs_to_print=print_epoch, skip_training=skip_train, setup_ddp=setup_ddp)

model_performance(trained_model, predictor_csv=train_csv, observed_csv=train_obsv, train_means=train_means,
                  train_stds=train_stds, obsv_mean=obsv_mean, obsv_std=obsv_std,
                  drop_columns=('fips', 'Year'))

