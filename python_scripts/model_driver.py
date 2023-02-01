from ML_ops import *
from download_preprocess import download_all_datasets, run_all_preprocessing, compile_observed_pumping_data

skip_download_gee_data = True
skip_download_ssebop_data = True
gee_data_list = ['MODIS_NDWI', 'GPM_PRECIP', 'MODIS_Day_LST', 'PRISM_PRECIP']
download_all_datasets(year_list=[2010, 2015], gee_data_list=gee_data_list,
                      skip_download_gee_data=skip_download_gee_data,
                      skip_download_ssebop_data=skip_download_ssebop_data)

exclude_data_from_df = ('MODIS_ET', 'MODIS_Terra_EVI', 'MODIS_Terra_NDVI', 'USDA_CDL')  # #
years = [2010, 2015]
skip_df_creation = True  # #
create_dataframe_csv(input_data_dir='../Data/Compiled_data',
                     output_csv='../Data/Compiled_data/predictor_csv/WesternUS_data.csv', search_by='*.tif',
                     years=years, drop_datasets=exclude_data_from_df, skip_dataframe_creation=skip_df_creation)


include_years = '*201[0-5]*.xlsx'  # #
skip_compiling = True  # #
compile_observed_pumping_data(data_dir='../Data/USGS_water_use_data', search_by=include_years,
                              county_shape='../Data/shapefiles/Western_US/WestUS_county_projected.shp',
                              output_csv='../Data/USGS_water_use_data/WestUS_county_gw_use.csv',
                              skip_compiling=skip_compiling)


data_frac, train_frac, val_frac, test_frac = 0.7, 0.7, 0.15, 0.15  # #
drop_data = None  # #
train_val_test_exists = False  # #
train_csv, validation_csv, test_csv, train_obsv, validation_obsv, test_obsv = \
    create_train_val_test_data(predictor_csv='../Data/Compiled_data/predictor_csv/WesternUS_data.csv',
                               observed_data_csv='../Data/USGS_water_use_data/WestUS_county_gw_use.csv',
                               data_fraction=data_frac, train_fraction=train_frac, val_fraction=val_frac,
                               test_fraction=test_frac, output_dir='../Data/Compiled_data/predictor_csv',
                               drop_columns=drop_data, train_val_test_exists=train_val_test_exists)

# Model training
hidden_layers = [100, 30, 10]  # #
activation = 'tanh'  # #
optim_method = 'adam'  # #
epoch = 3000  # #
rho = 0.01  # #
device = 'cuda'  # #
verbose = True  # #
skip_train = False  # #
trained_model = train_nn_model(predictor_csv=validation_csv, observed_csv=validation_obsv, hidden_layers=hidden_layers,
                               activation=activation, optimization=optim_method, epochs=epoch, learning_rate=rho,
                               device=device, verbose=verbose, skip_training=skip_train)



