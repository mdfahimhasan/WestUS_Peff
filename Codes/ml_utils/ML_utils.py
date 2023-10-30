import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor

from Codes.utils.system_ops import makedirs
from Codes.ml_utils.ML_model import reindex_df, calculate_r2, calculate_rmse, scatter_plot_of_same_vars
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster
from backup_codes.massP_pycnophylactic import run_mass_preserve, run_pycnophylactic_interp

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'



def create_prediction_raster(trained_model, training_csv, nan_pos_dict_path, prediction_raster, usgs_county_data,
                             drop_columns=['fips'], WaterUse_col_usgs_csv='gw_withdrawal',
                             ref_raster=WestUS_raster):
    # Ref raster load
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    # USGS county data
    usgs_df = pd.read_csv(usgs_county_data)
    usgs_df = usgs_df[['fips', WaterUse_col_usgs_csv]]

    # Training data load
    train_df = pd.read_csv(training_csv)
    fips_col = train_df[['fips']].to_numpy().squeeze()    # Saving fips to attach to the prediction for summing
    train_df = train_df.drop(columns=drop_columns)
    x_train = train_df.to_numpy()

    # Making model prediction using the trained model
    y_pred = trained_model.predict(x_train)

    # Loading nan_position_dict and replacing pixels of nan values with nodata/nan
    nan_pos_dict = pickle.load(open(nan_pos_dict_path, mode='rb'))

    for nan_pos in nan_pos_dict.values():
        y_pred[nan_pos] = ref_file.nodata

    # Creating output directory
    output_dir = os.path.dirname(prediction_raster)
    makedirs([output_dir])

    # Reshaping and writing raster
    y_pred_arr = y_pred.reshape(ref_arr.shape)
    write_array_to_raster(y_pred_arr, raster_file=ref_file, transform=ref_file.transform, output_path=prediction_raster)


def predict(trained_model, data_csv, output_dir, plot_name, drop_columns=['fips'], observed_data='totalW_observed'):

    df = pd.read_csv(data_csv)

    df = df.drop(columns=drop_columns)
    x = df.drop(columns=observed_data).to_numpy()
    y_obsv = df[[observed_data]].to_numpy().squeeze()

    y_pred = trained_model.predict(x)

    print(f'calculate_r2 score= {calculate_r2(y_pred, y_obsv)}')
    print(f'calculate_rmse score= {calculate_rmse(y_pred, y_obsv)}')
    scatter_plot_of_same_vars(y_pred, y_obsv, plot_name=plot_name, savedir=output_dir)