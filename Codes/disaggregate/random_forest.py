import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor

from Codes.utils.system_ops import makedirs
from Codes.ML_ops import reindex_df, calculate_r2, calculate_rmse, scatter_plot
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster
from Codes.disaggregate.massP_pycnophylactic import run_mass_preserve, run_pycnophylactic_interp

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def apply_LandUse_filter(crop_raster, developed_raster, threshold_percent=0.03,
                         county_area_csv='../../Data_main/USGS_water_use_data/US_county_area.csv',
                         countyID_raster='../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif',
                         county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county.shp',
                         ref_raster=WestUS_raster):
    county_area_df = pd.read_csv(county_area_csv)

    ref_arr = read_raster_arr_object(ref_raster, get_file=False)
    crop_arr = read_raster_arr_object(crop_raster, get_file=False)
    developed_arr = read_raster_arr_object(developed_raster, get_file=False)
    landUse_arr = np.where((crop_arr > 0) | (developed_arr > 0), 1, ref_arr)

    county_arr = read_raster_arr_object(countyID_raster, get_file=False)
    unique_fips, counts = np.unique(county_arr[~np.isnan(county_arr) & (landUse_arr == 1)], return_counts=True)
    count_dict = {'fips': unique_fips, 'pixel_counts': counts}

    # Creating a dictionary which have counts of cropland (irrigated) and developed pixels in each county
    count_df = pd.DataFrame(count_dict)
    area_single_pixel = (2.22 * 2.22) * (1000 * 1000)  # area of a pixel in m2
    count_df['pixel_area'] = count_df['pixel_counts'] * area_single_pixel

    county_gdf = gpd.read_file(county_shape)
    county_gdf = county_gdf.merge(count_df, on=['fips'], how='left')  # 1st merge to add pixel area information
    county_gdf = county_gdf.drop(columns=['COUNTYNS', 'AFFGEOID', 'GEOID', 'NAME', 'LSAD', 'ALAND', 'AWATER', ])
    county_gdf = county_gdf.merge(county_area_df, on=['fips'], how='left')
    county_gdf = county_gdf.drop(columns=['COUNTYNS', 'OBJECTID', ])  # 2nd merge to add total county area information

    # Sorting out countries with irrigated and developed area lower that threshold county area
    # ALAND is total county area in m2
    sorted_counties = county_gdf[count_df['pixel_area'] < (threshold_percent * county_gdf['ALAND'])]
    remove_fips = list(sorted_counties['fips'])

    county_gdf['used_in_T'] = np.where(~county_gdf['fips'].isin(remove_fips), 'used', 'not used')
    county_gdf.to_file('../../scratch/counties_TTV.shp')

    return remove_fips, county_gdf


def create_dataframe_csv(input_data_dir, output_dir, water_use_raster, search_by='*.tif', years=[2015],
                         drop_datasets=('MODIS_ET', 'MODIS_Terra_EVI', 'MODIS_Terra_NDVI'),
                         encode_cols=('USDA_cropland', 'USDA_developed'),
                         skip_dataframe_creation=False):
    """
    Create dataframe from predictor variables (raster).

    ** All variable names should start with capital letter. This is needed for keeping fips to the last during
       reindex().

    :param input_data_dir: Input rasters' directory.
    :param output_dir: Output directory filepath.
    :param search_by: Input raster search criteria. Defaults to '*.tif'.
    :param years: Tuple/List of years to be included in predictor dataset. Default set to [2015].
    :param drop_datasets: Tuple/List of variable names to drop while creating dataframe/csv.
    :param encode_cols: Tuple/List of variable names that need one hot encoding. Set to None if no encoding is required.
    :param skip_dataframe_creation: Set to True if want to skip processing.

    :return: A dataframe created where columns represent individual predictors.
    """
    makedirs([output_dir])

    # Code block will be modified here for static predictors (we might/might not need timeframe at all)
    county_id_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif'

    if not skip_dataframe_creation:
        print('Creating Predictors csv...')
        predictors_list = glob(os.path.join(input_data_dir, search_by))
        predictors = []

        # Applying yearly filter
        try:
            years_str = [str(y) for y in list(years)]  # if processing data of multiple years
        except:
            years_str = [str(years)]  # if processing data of a single year

        for prd in predictors_list:  # to add predictors of only selected years
            if any(yr in prd for yr in years_str):
                predictors.append(prd)

        predictors = sorted(predictors)

        training_variable_dict = {}  # used during model training
        prediction_variable_dict = {}  # used to create model prediction with trained model
        nan_position_dict = {}  # dictionary to store nan positions for variables. Used to make final prediction raster

        for year in years_str:
            for predictor in predictors:
                name = os.path.splitext(os.path.basename(predictor))[0]
                yr = name[name.rfind('_') + 1:]
                if yr == year:
                    variable_name = name[:name.rfind('_')]

                    if variable_name not in drop_datasets:  # predictors in drop_datasets will not be in predictor csv
                        # Processing predictor data for model training
                        raster_arr = read_raster_arr_object(predictor, get_file=False)
                        raster_arr = raster_arr.flatten()
                        training_variable_dict[variable_name] = raster_arr

                        # Processing predictor data for model prediction
                        nan_position_dict[variable_name] = np.isnan(raster_arr)
                        raster_arr_for_pred = raster_arr
                        raster_arr_for_pred[nan_position_dict[variable_name]] = 0
                        prediction_variable_dict[variable_name] = raster_arr_for_pred
                else:
                    pass  # for data that doesn't belong to a specific year in each yearly iteration

            # Adding county ID data
            id_arr = read_raster_arr_object(county_id_raster, get_file=False)
            id_arr = id_arr.flatten()
            training_variable_dict['countyID'] = id_arr
            prediction_variable_dict['countyID'] = id_arr

            # Adding the disaggregated "Mass Preserved Pycnophylactic" observed data
            wateruse_obsv_arr = read_raster_arr_object(water_use_raster, get_file=False)
            wateruse_obsv_arr = wateruse_obsv_arr.flatten()
            training_variable_dict['wateruse_observed'] = wateruse_obsv_arr

            # Don't need nan positions for observed data it will not be used as predictors

            var_df_for_train = pd.DataFrame(training_variable_dict)
            var_df_for_train = var_df_for_train.dropna(axis='index')

            var_df_for_prediction = pd.DataFrame(prediction_variable_dict)

            # One-hot encoding
            if encode_cols is not None:
                for col in encode_cols:
                    var_df_enc = pd.get_dummies(var_df_for_train[col], prefix=col)
                    pred_df_enc = pd.get_dummies(var_df_for_prediction[col], prefix=col)
                    for i in var_df_enc.columns:    # adding encoded columns to the main dataframe
                        var_df_for_train[i] = var_df_enc[i]
                        var_df_for_prediction[i] = pred_df_enc[i]

            # Saving dataframe to be used for model training
            var_df_for_train = var_df_for_train.rename(columns={'countyID': 'fips'})
            var_df_for_train = reindex_df(var_df_for_train)
            # var_df_for_train = var_df_for_train.sort_values(by=['fips'], ascending=True, axis=0)

            csv_name = f'variables_train_{year}'
            path = os.path.join(output_dir, f'{csv_name}.csv')
            var_df_for_train.to_csv(path, index=False)

            # Saving dataframe to be used for model prediction
            var_df_for_prediction = var_df_for_prediction.rename(columns={'countyID': 'fips'})
            var_df_for_prediction = reindex_df(var_df_for_prediction)
            # var_df_for_prediction = var_df_for_prediction.sort_values(by=['fips'], ascending=True, axis=0)

            csv_name = f'variables_prediction_{year}'
            path = os.path.join(output_dir, f'{csv_name}.csv')
            var_df_for_prediction.to_csv(path, index=False)

            nan_dict_path = f'../../Data_main/Model_csv/rf_gw/nan_dict_{year}.pkl'
            pickle.dump(nan_position_dict, open(nan_dict_path, mode='wb+'))

        print('Predictors csv created\n')

    else:
        print('Predictors csv exists\n')

    return output_dir


def process_df_with_LU_filter(input_csv, output_csv, crop_raster, developed_raster, threshold_percent=0.03,
                              drop_columns=None, skip_dataframe_creation=False):
    if not skip_dataframe_creation:
        print('Filtering counties for training...\n')
        df = pd.read_csv(input_csv)

        remove_fips, _ = apply_LandUse_filter(crop_raster=crop_raster, developed_raster=developed_raster,
                                              threshold_percent=threshold_percent)
        filtered_df = df[~df['fips'].isin(remove_fips)]
        if drop_columns is not None:
            filtered_df = filtered_df.drop(columns=drop_columns)

        filtered_df.to_csv(output_csv, index=False)
    else:
        print('Filtered training data exists \n')
        pass
    return output_csv


def create_train_val_test_data(predictor_csv, output_dir, crop_raster, developed_raster,
                               data_fraction=0.8, train_fraction=0.7, val_fraction=0.15,
                               test_fraction=0.15, threshold_landuse=0.05,
                               drop_columns=None, train_val_test_exists=False):
    """
    Create Train, Validation, and Test dataset (csv) from the predictor csv.

    :param predictor_csv: File path of predictor csv as input.
    :param observed_data_csv: File path of observed county-wise pumping data csv as input.
    :param data_fraction: Fraction of data to be used for model training, validating, and testing.
    :param train_fraction: Fraction of data (from use_data_fraction of data_csv) to be used for training.
    :param val_fraction: Fraction of data (from use_data_fraction of data_csv) to be used for validating.
    :param test_fraction: Fraction of data (from use_data_fraction of data_csv) to be used for testing.
    :param output_dir: File path of output directory to save/load train, validation, and test datasets.
    :param drop_columns: List of columns to drop from the predictor csv. Default set to None.
    :param train_val_test_exists: Set to True if want to skip train, validation, and test data creation again.
                                  Existing datasets will be loaded.

    :return: Train, Validation, and Test dataset variables and observed csv.
    """
    if not train_val_test_exists:
        print('Creating Train, Validation, Test dataset...')

        # Creating fips_unique and selecting fips that will be splitted into train, validation, and
        # test data.
        # # Processing predictor dataset
        predictor_df = pd.read_csv(predictor_csv)
        if drop_columns is not None:
            predictor_df = predictor_df.drop(Columns=drop_columns, axis=1)

        remove_fips_LandUse_threshold = apply_LandUse_filter(crop_raster=crop_raster, developed_raster=developed_raster,
                                                             threshold_percent=threshold_landuse)
        print(f'Using landuse threshold to filter counties: {threshold_landuse}')

        fips_unique = predictor_df['fips'].unique()
        fips_unique = list(np.random.permutation(fips_unique))
        fips_unique = [i for i in fips_unique if i not in remove_fips_LandUse_threshold]

        print('Number of counties after landuse filter:', len(fips_unique), '\n')

        train_length = round(len(fips_unique) * data_fraction * train_fraction)
        validation_length = round(len(fips_unique) * data_fraction * val_fraction)
        test_length = round(len(fips_unique) * data_fraction * test_fraction)

        # Selecting fips based on lengths
        train_fips = fips_unique[:train_length]
        validation_fips = fips_unique[train_length:(train_length + validation_length)]
        test_fips = fips_unique[
                    (train_length + validation_length):(train_length + validation_length + test_length)]

        # Top counties with high pumping has been added to the train, validation, and test fips. This is a very
        # important step for train dataset, to ensure that high pumping counties are included in model training
        predictor_df = predictor_df.sort_values(by='wateruse_observed', ascending=False)
        top_pumping_counties = predictor_df.iloc[0:200, :]['fips'].to_list()
        top_pumping_counties = list(np.random.permutation(top_pumping_counties))
        top_pumping_counties = [i for i in top_pumping_counties if i not in remove_fips_LandUse_threshold]

        top_train_length = round(len(top_pumping_counties) * train_fraction)
        top_val_length = round(len(top_pumping_counties) * val_fraction)
        top_test_length = round(len(top_pumping_counties) * test_fraction)

        top_train_fips = top_pumping_counties[:top_train_length]
        top_val_fips = top_pumping_counties[top_train_length:(top_train_length + top_val_length)]
        top_test_fips = top_pumping_counties[(top_train_length + top_val_length):
                                             (top_train_length + top_val_length + top_test_length)]

        # Modified train, validation, and test fips with added fips with high pumping values
        train_fips = train_fips + [i for i in top_train_fips if i not in (train_fips + validation_fips + test_fips)]
        validation_fips = validation_fips + \
                          [i for i in top_val_fips if i not in (train_fips + validation_fips + test_fips)]
        test_fips = test_fips + [i for i in top_test_fips if i not in (train_fips + validation_fips + test_fips)]

        predictor_df = predictor_df.drop(columns=['Year'])
        predictor_df = predictor_df.sort_values(by=['fips'], ascending=True, axis=0)

        train_set = predictor_df[predictor_df['fips'].isin(train_fips)]
        validation_set = predictor_df[predictor_df['fips'].isin(validation_fips)]
        test_set = predictor_df[predictor_df['fips'].isin(test_fips)]

        train_set= train_set.sort_values(by=['fips'], ascending=True, axis=0)
        validation_set = validation_set.sort_values(by=['fips'], ascending=True, axis=0)
        test_set = test_set.sort_values(by=['fips'], ascending=True, axis=0)

        train_set.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
        validation_set.to_csv(os.path.join(output_dir, 'validation_set.csv'), index=False)
        test_set.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)

        print('Train, Validation, Test datasets created')

        train_csv = os.path.join(output_dir, 'train_set.csv')
        val_csv = os.path.join(output_dir, 'validation_set.csv')
        test_csv = os.path.join(output_dir, 'test_set.csv')

    else:
        print('Loading Train/val/test data... \n')
        train_csv = os.path.join(output_dir, 'train_set.csv')
        val_csv = os.path.join(output_dir, 'validation_set.csv')
        test_csv = os.path.join(output_dir, 'test_set.csv')

    return train_csv, val_csv, test_csv,


def run_RF(train_csv, drop_columns=['fips'], observed_data='wateruse_observed', number_of_runs=10,
           usgs_county_data='../../Data_main/USGS_water_use_data/WestUS_county_WaterUse.csv',
           WaterUse_col_usgs_csv='gw_withdrawal', plot_dir='../../Model_Run/Plots'):
    # USGS county data
    usgs_df = pd.read_csv(usgs_county_data)
    usgs_df = usgs_df[['fips', WaterUse_col_usgs_csv]]

    # Training data load
    train_df = pd.read_csv(train_csv)
    fips_col = train_df[['fips']].to_numpy().squeeze()    # Saving fips to attach to the prediction for summing
    train_df = train_df.drop(columns=drop_columns)

    x_train = train_df.drop(columns=observed_data).to_numpy()
    y_train = train_df[[observed_data]].to_numpy().squeeze()

    county_rmse_track = []
    for i in range(1, number_of_runs+1):
        rf_regression_model = RandomForestRegressor(n_estimators=200, max_features=4, max_depth=20,
                                                    random_state=0, bootstrap=True,
                                                    n_jobs=-1, oob_score=True)

        trained_model = rf_regression_model.fit(x_train, y_train)
        y_pred = trained_model.predict(x_train)

        # # updating training data
        y_pred_stacked = np.hstack((fips_col.reshape(-1, 1), y_pred.reshape(-1, 1)))
        y_update_df = pd.DataFrame(y_pred_stacked, columns=['fips', 'y_pred'])
        grouped_y_pred_df = y_update_df.groupby(by=['fips'])['y_pred'].sum().reset_index()
        grouped_y_pred_df.columns = ['fips', 'y_pred_sum']

        # 1st merge: merging summed predicted data with USGS county data
        grouped_y_pred_df = grouped_y_pred_df.merge(usgs_df, on='fips', how='left')

        # Calculating ratio for y_train updating

        ########## Think critically about this. Whether this formula is good or "grouped_y_pred_df[WaterUse_col_usgs_csv]"
        ########## will change in each iteration
        grouped_y_pred_df['ratio'] = grouped_y_pred_df[WaterUse_col_usgs_csv] / grouped_y_pred_df['y_pred_sum']

        # 2nd merge: merging the grouped df with pixel-wise y_update_df to transfer ratio information
        y_update_df = y_update_df.merge(grouped_y_pred_df, on='fips', how='left')
        y_update_df['y_train_updated'] = y_update_df['y_pred'] * y_update_df['ratio']

        # # Performance
        # Pixel-level score
        print(f'### Model iteration: {i} ###')
        # print(f'pixel R2: {calculate_r2(y_pred, y_train)}')
        # print(f'pixel RMSE: {calculate_rmse(y_pred, y_train)}')

        # county-level score
        predicted_at_county = grouped_y_pred_df['y_pred_sum']
        observed_at_county = grouped_y_pred_df[WaterUse_col_usgs_csv]
        county_r2 = calculate_r2(Y_pred=predicted_at_county, Y_obsv=observed_at_county)  # R2
        county_rmse = calculate_rmse(Y_pred=predicted_at_county, Y_obsv=observed_at_county)  # RMSE
        county_rmse_track.append(county_rmse)  # a list to track county level rmse

        print(f'county R2: {county_r2}')
        print(f'county RMSE: {county_rmse}\n')
        if i == number_of_runs:
            scatter_plot(Y_pred=predicted_at_county, Y_obsv=observed_at_county, plot_name='R2_plot_county.jpeg',
                         savedir=plot_dir)

        # # Formatting training data for next iteration
        y_train = y_update_df[['y_train_updated']].to_numpy().squeeze()

    y_train_final = y_train

    return trained_model, y_train_final


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
    scatter_plot(y_pred, y_obsv, plot_name=plot_name, savedir=output_dir)


# Shapefile to Water Use Raster
convert_WaterUse_shp_to_raster = False  # #
if convert_WaterUse_shp_to_raster:
    shapefile_to_raster(input_shape='../../Data_main/Compiled_data/County_total_WaterUse.shp',
                        output_dir='../../scratch',
                        raster_name='WestUS_obsv_total_Use_2km.tif', use_attr=True, attribute='total_with', add=None,
                        ref_raster=WestUS_raster, resolution=model_res, alltouched=False)

    shapefile_to_raster(input_shape='../../Data_main/Compiled_data/County_total_WaterUse.shp',
                        output_dir='../../scratch',
                        raster_name='WestUS_obsv_sw_use_2km.tif', use_attr=True, attribute='sw_withdra', add=None,
                        ref_raster=WestUS_raster, resolution=model_res, alltouched=False)

    shapefile_to_raster(input_shape='../../Data_main/Compiled_data/County_total_WaterUse.shp',
                        output_dir='../../scratch',
                        raster_name='WestUS_obsv_gw_use_2km.tif', use_attr=True, attribute='gw_withdra', add=None,
                        ref_raster=WestUS_raster, resolution=model_res, alltouched=False)

run_mass_pynco = False  # #
if run_mass_pynco:
    # Datasets
    crop_raster = '../../Data_main/Compiled_data/Irrigated_agri_2015.tif'
    # crop_raster = '../../Data_main/Compiled_data/USDA_cropland_2015.tif'
    developed_raster = '../../Data_main/Compiled_data/USDA_developed_2015.tif'
    countyID_raster = '../../Data_main/Compiled_data/Western_US_countyID.tif'

    WaterUse_raster = '../../scratch/WestUS_obsv_gw_Use_2km.tif'
    output_massP_raster = '../../scratch/mass_preserved_gw.tif'

    # mass preserving
    massP_raster = run_mass_preserve(countyID_raster, WaterUse_raster, crop_raster=crop_raster,
                                     developed_raster=developed_raster, output_MassPreserved_file=output_massP_raster)

    # pycnophylactic
    final_pycno_raster = '../../scratch/pycno_gw.tif'
    run_pycnophylactic_interp(massP_raster, WaterUse_raster, countyID_raster, final_pycno_raster)

skip_df_creation = True
create_dataframe_csv(input_data_dir='../../Data_main/Compiled_data', output_dir='../../Data_main/Model_csv/rf_gw',
                     water_use_raster='../../scratch/pycno_gw.tif', search_by='*.tif', years=[2015],
                     drop_datasets=('MODIS_ET', 'MODIS_Terra_EVI', 'MODIS_Terra_NDVI'),
                     encode_cols=None, skip_dataframe_creation=skip_df_creation)

skip_df_filtering = True
training_csv = process_df_with_LU_filter(input_csv='../../Data_main/Model_csv/rf_gw/variables_train_2015.csv',
                                         output_csv='../../Data_main/Model_csv/rf_gw/filtered_variables_train_2015.csv',
                                         crop_raster='../../Data_main/Compiled_data/irrigated_agri_2015.tif',
                                         developed_raster='../../Data_main/Compiled_data/USDA_developed_2015.tif',
                                         threshold_percent=0.02,
                                         drop_columns=None, skip_dataframe_creation=skip_df_filtering)


# Model training
trained_rf, disagg_gw_use = run_RF(train_csv=training_csv, drop_columns=['fips'], observed_data='wateruse_observed',
                                   number_of_runs=10,
                                   usgs_county_data='../../Data_main/USGS_water_use_data/WestUS_county_WaterUse.csv',
                                   WaterUse_col_usgs_csv='gw_withdrawal')

create_prediction_raster(trained_model=trained_rf, training_csv='../../Data_main/Model_csv/rf_gw/variables_prediction_2015.csv',
                         nan_pos_dict_path='../../Data_main/Model_csv/rf_gw/nan_dict_2015.pkl',
                         prediction_raster='../../Model_Run/Prediction/prediction_gw.tif',
                         usgs_county_data='../../Data_main/USGS_water_use_data/WestUS_county_WaterUse.csv',
                         drop_columns=['fips'], WaterUse_col_usgs_csv='gw_withdrawal', ref_raster=WestUS_raster)

#
#
# train_val_test_data_exists = True  # #
# data_frac, train_frac, val_frac, test_frac = 0.8, 0.7, 0.15, 0.15  # #
# threshold_landuse = 0.04  # #
# train_csv, val_csv, test_csv = \
#     create_train_val_test_data(predictor_csv='../../scratch/rf_gw/predictor_df.csv',
#                                threshold_landuse=threshold_landuse, data_fraction=data_frac,
#                                train_fraction=train_frac, val_fraction=val_frac, test_fraction=test_frac,
#                                output_dir='../../scratch/rf_gw', drop_columns=None,
#                                crop_raster='../../Data_main/Compiled_data/USDA_cropland_2015.tif',
#                                developed_raster='../../Data_main/Compiled_data/USDA_developed_2015.tif',
#                                train_val_test_exists=train_val_test_data_exists)
#

# # Training performance
# print('Training Performance -')
# output_dir = '../../scratch/rf_gw'
# predict(trained_model=trained_rf, data_csv=train_csv, output_dir=output_dir, plot_name='trainedR2.jpeg',
#         drop_columns=('fips'), observed_data='wateruse_observed')


# # Testing performance
# print('Testing Performance -')
# output_dir = '../../scratch/rf_gw'
# predict(trained_model=trained_rf, data_csv=test_csv, output_dir=output_dir, plot_name='testR2.jpeg',
#         drop_columns=('fips'), observed_data='wateruse_observed')
#
#
# # Validation performance
# print('Validation Performance -')
# output_dir = '../../scratch/rf_gw'
# predict(trained_model=trained_rf, data_csv=val_csv, output_dir=output_dir, plot_name='valR2.jpeg',
#         drop_columns=('fips'), observed_data='wateruse_observed')