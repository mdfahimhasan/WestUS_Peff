import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.data_download_preprocess.preprocesses import sum_cropET_water_yr, dynamic_gs_sum_peff
from Codes.effective_precip.m00_eff_precip_utils import estimate_peff_precip_water_year_fraction, \
    estimate_water_yr_peff_using_peff_frac, scale_monthy_peff_with_wateryr_peff_model


# # # Steps

# Step 1: water year peff raster creation using water year peff fraction (water year precip * water year fraction)
# Step 2: scaling monthly peff prediction with annual model
# Step 3: sum scaled monthly peff to water year (for model check)
# Step 4: estimate water year peff/precipitation fraction (for model check)
# Step 5: sum scaled monthly peff to growing season (with added 3 months' peff before growing season to consider carried over soil moisture storage)

if __name__ == '__main__':
    model_version = 'v15'                           #####
    skip_estimating_peff_water_yr_total = False      #####
    skip_peff_monthly_scaling = False                #####
    skip_sum_scaled_peff_water_year = False          #####
    skip_peff_frac_estimate_water_yr = False         #####
    skip_sum_scale_peff_to_gs = False                #####

    # # # # # Step 1: water year peff raster creation using water year peff fraction # # # # #
    years = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
             2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
             2016, 2017, 2018, 2019, 2020)
    water_year_precip_dir = '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year'
    water_year_peff_frac_dir = f'../../Data_main/Raster_data/Effective_precip_fraction_WestUS/{model_version}_water_year_frac'
    output_updated_peff_water_yr_dir = f'../../Data_main/Raster_data/Effective_precip_fraction_WestUS/{model_version}_water_year_total_from_fraction'

    estimate_water_yr_peff_using_peff_frac(years, water_year_precip_dir, water_year_peff_frac_dir,
                                           output_updated_peff_water_yr_dir,
                                           skip_processing=skip_estimating_peff_water_yr_total)

    # # # # #  Step 2: scaling monthly peff prediction with annual model # # # # #
    years = (1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
             2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
             2016, 2017, 2018, 2019, 2020)
    unscaled_peff_monthly_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
    unscaled_peff_water_yr_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_water_year'
    peff_monthly_scaled_output_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly_scaled'

    scale_monthy_peff_with_wateryr_peff_model(years, unscaled_peff_monthly_dir, unscaled_peff_water_yr_dir,
                                              output_updated_peff_water_yr_dir, peff_monthly_scaled_output_dir,
                                              skip_processing=skip_peff_monthly_scaling)

    # # # # # Step 3: summing scaled monthly effective precipitation for water year # # # # #
    output_peff_scaled_water_year_summed_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_water_year_scaled'

    sum_cropET_water_yr(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                    2016, 2017, 2018, 2019, 2020),
                        input_cropET_monthly_dir=peff_monthly_scaled_output_dir,
                        output_dir_water_yr=output_peff_scaled_water_year_summed_dir,
                        save_keyword='effective_precip',
                        skip_processing=skip_sum_scaled_peff_water_year)

    # # # # #  Step4: estimating water year peff fraction for scaled peff data # # # # #
    scaled_frac_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_peff_fraction_scaled'

    estimate_peff_precip_water_year_fraction(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                                         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                                         2016, 2017, 2018, 2019, 2020),
                                             peff_dir_water_yr=output_peff_scaled_water_year_summed_dir,
                                             precip_dir_water_yr=water_year_precip_dir,
                                             output_dir=scaled_frac_dir,
                                             skip_processing=skip_peff_frac_estimate_water_yr)

    # # # # #  Step 5: compile scale monthly Peff to growing season # # # # #
    output_peff_grow_season_summed_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season_scaled'

    dynamic_gs_sum_peff(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                   2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                   2016, 2017, 2018, 2019),
                        growing_season_dir='../../Data_main/Raster_data/Growing_season',
                        monthly_input_dir=peff_monthly_scaled_output_dir,
                        gs_output_dir=output_peff_grow_season_summed_dir,
                        sum_keyword='effective_precip',
                        skip_processing=skip_sum_scale_peff_to_gs)


# def classify_irrigated_rainfed_cropland(irrigated_fraction_dir, irrigated_cropland_output_dir,
#                                         skip_processing=False):
#     """
#     Classifies rainfed and irrigated cropland using rainfed and irrigated fraction data.
#
#     ** The rainfed fraction data is only available for 2008-2020, as CDL has Western US scale crop classification data
#     starting from 2008. This function classifies rainfed cropland data for 2008-2020 only, but irrigated cropland data
#     for 2000-2020. **
#
#     :param rainfed_fraction_dir: Input directory path for rainfed fraction data.
#     :param irrigated_fraction_dir: Input directory path for irrigated fraction data.
#     :param tree_cover_dir: Input directory for tree cover dataset.
#     :param rainfed_cropland_output_dir: Output directory path for classified rainfed cropland data.
#     :param irrigated_cropland_output_dir: Output directory path for classified irrigated cropland data.
#     :param skip_processing: Set to True if want to skip classifying irrigated and rainfed cropland data.
#
#     :return: None
#     """
#     if not skip_processing:
#         makedirs([irrigated_cropland_output_dir])
#
#         ############################
#         # # Irrigated
#         # A 2km pixel with any fraction of irrigation will be classified as irrigated.
#         # Previously, >2% irr fraction was used to classify as irrigated, which was later removed to cover the
#         # boundary pixels in irrigated-agricultural zones. May need post filtering.
#         irrigated_frac_threshold_for_irrigated_class = 0.02
#
#         years_rest_irrigated_frac_data = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
#                                                        2015, 2016, 2017, 2018, 2019, 2020]
#
#         for year in years_rest_irrigated_frac_data:
#             print(f'Classifying rainfed and irrigated cropland data for year {year}')
#
#             irrigated_frac_data = os.path.join(irrigated_fraction_dir, f'Irrigated_Frac_{year}.tif')
#
#             irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)
#
#             # classification using defined rainfed and irrigated fraction threshold. -9999 is no data
#             irrigated_cropland = np.where(irrig_arr > irrigated_frac_threshold_for_irrigated_class, 1, -9999)
#
#             # saving classified data
#             output_irrigated_cropland_raster = os.path.join(irrigated_cropland_output_dir, f'Irrigated_cropland_{year}.tif')
#             write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file, transform=irrig_file.transform,
#                                   output_path=output_irrigated_cropland_raster, dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
#     else:
#         pass
#
#
# def filter_rainfed_irrigated_cropET_with_rainfed_irrigated_cropland(irrigated_cropland_dir,
#                                                                     irrigated_cropET_input_dir,
#                                                                     irrigated_cropET_output_dir,
#                                                                     peff_input_dir,
#                                                                     peff_output_dir,
#                                                                     skip_processing=False):
#     """
#     Filter Irrigated and Rainfed cropET data by rainfed and irrigated cropland, respectively.
#
#     ** The downloaded Irrigated and Rainfed cropET data from GEE is not fully filtered for rainfed and irrigated
#     cropland because in some pixels there are some rainfed and some irrigated fields. So, we first classify rainfed and
#     irrigated cropland by rainfed and irrigation fraction threshold (detail in classify_irrigated_rainfed_cropland()
#     func), then apply the cropland filters to extract cropET on the purely rainfed and irrigated cropland pixels.
#
#     :param rainfed_cropland_dir: Input directory filepath of rainfed cropland data.
#     :param irrigated_cropland_dir: Input directory filepath of irrigated cropland data.
#     :param rainfed_cropET_input_dir: Input directory filepath of raw rainfed cropET data.
#     :param irrigated_cropET_input_dir: Input directory filepath of raw irrigated cropET data.
#     :param rainfed_cropET_output_dir: Output directory filepath of filtered rainfed cropET data.
#     :param irrigated_cropET_output_dir: Output directory filepath of filtered irrigated cropET data.
#     :param skip_processing: Set to True if want to skip filtering irrigated and rainfed cropET data.
#
#     :return: None.
#     """
#     if not skip_processing:
#         makedirs([irrigated_cropET_output_dir, peff_output_dir])
#         # cropET datasets have been extracted from openET for the following years_list and months only
#         years_to_filter_irrig_cropET = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
#                                         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
#                                         2016, 2017, 2018, 2019, 2020]
#
#         months_to_filter_cropET = list(range(1, 13))
#
#         for year in years_to_filter_irrig_cropET:
#             print(f'Filtering irrigated cropET and Peff data for year {year}...')
#
#             # pure irrigated cropland filtered previously by using irrigated fraction threshold (irrig frac > 0.20)
#             irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
#             irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)
#
#             peff_data = glob(os.path.join(peff_input_dir, f'*{year}*.tif'))[0]
#             peff_arr, file = read_raster_arr_object(peff_data)
#
#             # applying the filter
#             peff_arr[np.isnan(irrigated_cropland_arr)] = -9999
#             output_raster = os.path.join(peff_output_dir, f'effective_precip_{year}.tif')
#             write_array_to_raster(raster_arr=peff_arr, raster_file=file,
#                                   transform=file.transform, output_path=output_raster)
#
#             for month in months_to_filter_cropET:
#                 # # applying irrigated cropland filter to get cropET at purely irrigated pixels
#                 irrigated_cropET_data = glob(os.path.join(irrigated_cropET_input_dir, f'*{year}_{month}*.tif'))[0]
#                 irrigated_cropET_arr, irrigated_cropET_file = read_raster_arr_object(irrigated_cropET_data)
#
#                 # applying the filter
#                 irrigated_cropET_arr[np.isnan(irrigated_cropland_arr)] = -9999
#
#                 filtered_output_raster = os.path.join(irrigated_cropET_output_dir, f'Irrigated_cropET_{year}_{month}.tif')
#                 write_array_to_raster(raster_arr=irrigated_cropET_arr, raster_file=irrigated_cropET_file,
#                                       transform=irrigated_cropET_file.transform, output_path=filtered_output_raster)
#     else:
#         pass

# # classify rainfed and irrigated cropland data
# classify_irrigated_rainfed_cropland(
#     irrigated_fraction_dir='../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac',
#     irrigated_cropland_output_dir='../../Data_main/Raster_data/Irrigated_cropland_v2',
#     skip_processing=True)
#
# # sum monthly irrigated cropET for dynamic growing season
# dynamic_gs_sum_ET(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
#                              2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
#                              2016, 2017, 2018, 2019, 2020),
#                   growing_season_dir='../../Data_main/Raster_data/Growing_season',
#                   monthly_input_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly_v2',
#                   gs_output_dir='../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season_v2',
#                   sum_keyword='Irrigated_cropET',
#                   skip_processing=True)



# # checking water year Peff/precipitation fraction for filtered training data
# monthly_filtered_training_data = '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/final_filtered_cropET_for_training'
# output_water_year_summed_training_data = '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/final_filtered_cropET_for_training_water_year'
#
# sum_cropET_water_yr(years_list=(2009, 2010, 2011, 2012, 2013, 2014,
#                                         2015, 2016, 2017, 2018, 2019, 2020),
#                             input_cropET_monthly_dir=monthly_filtered_training_data,
#                             output_dir_water_yr=output_water_year_summed_training_data,
#                             skip_processing=True)
#

# precip_dir_water_yr = '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year'
# frac_dir = '../../Data_main/Raster_data/Rainfed_cropET_filtered_training/rainfed_cropET_fraction'
#
# estimate_peff_precip_water_year_fraction(years_list=(2009, 2010, 2011, 2012, 2013, 2014,
#                                                     2015, 2016, 2017, 2018, 2019, 2020),
#                                          peff_dir_water_yr=output_water_year_summed_training_data,
#                                          precip_dir_water_yr=precip_dir_water_yr,
#                                          output_dir=frac_dir,
#                                          skip_processing=True)
#
#
# rainfed_cropET_water_year = r'F:\WestUS_NetGW\Data_main\Raster_data\Rainfed_cropET\WestUS_water_year'
# precip_dir_water_yr = '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year'
# frac_dir = '../../Data_main/Raster_data/Rainfed_cropET/rainfed_cropET_fraction'
#
# estimate_peff_precip_water_year_fraction(years_list=(2009, 2010, 2011, 2012, 2013, 2014,
#                                                     2015, 2016, 2017, 2018, 2019, 2020),
#                                          peff_dir_water_yr=rainfed_cropET_water_year,
#                                          precip_dir_water_yr=precip_dir_water_yr,
#                                          output_dir=frac_dir,
#                                          skip_processing=True)



# # counting how many pixels get water_year precip < water_year Peff and vice versa from the original rainfed training data over pasture
# rainfed_cropET = r'F:\WestUS_NetGW\Data_main\Raster_data\Rainfed_cropET\WestUS_water_year\Rainfed_cropET_2015.tif'
# precip = r'F:\WestUS_NetGW\Data_main\Raster_data\GRIDMET_Precip\WestUS_water_year\GRIDMET_Precip_water_yr_2015.tif'
# filtered_training_data = r'F:\WestUS_NetGW\Data_main\Raster_data\Rainfed_cropET_filtered_training\final_filtered_cropET_for_training\CropET_for_training_2019_4.tif'
#
# rain_arr = read_raster_arr_object(rainfed_cropET, get_file=False)
# precip_arr = read_raster_arr_object(precip, get_file=False)
# filtered_arr = read_raster_arr_object(filtered_training_data, get_file=False)
#
# total_cropET_pixels = np.count_nonzero(np.where(~np.isnan(rain_arr), 1, 0))
# valid_train_pixels = np.count_nonzero(np.where(~np.isnan(rain_arr) & (precip_arr > rain_arr), 1, 0))
# filtered_training_pixels = np.count_nonzero(np.where(~np.isnan(filtered_arr), 1, 0))
#
#
# print(f'rainfed cropET pixels: {total_cropET_pixels}')
# print(f'water yr precip > water yr rainfed cropET pixels: {valid_train_pixels}')
# print(f'water yr precip > water yr rainfed cropET pixel %: {valid_train_pixels * 100/ total_cropET_pixels}')
# print(f'Filtered training rainfed CropET pixels in a month: {filtered_training_pixels}')
# print(f'Filtered training rainfed CropET pixels as a percentage of valid rainfed cropET pixels: '
#       f'{filtered_training_pixels * 100 / valid_train_pixels}')