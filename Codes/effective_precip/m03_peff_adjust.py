import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.data_download_preprocess.preprocesses import sum_cropET_water_yr, dynamic_gs_sum_peff_with_3m_SM_storage, \
    dynamic_gs_sum_ET
from Codes.effective_precip.m00_eff_precip_utils import estimate_peff_precip_water_year_fraction, \
    estimate_water_yr_peff_using_peff_frac, scale_monthy_peff_with_wateryr_peff_model

# # # Steps

# Step 1: water year peff raster creation using water year peff fraction (water year precip * water year fraction)
# Step 2: scaling monthly peff prediction with water year model
# Step 3: sum scaled monthly peff to water year (for model check)
# Step 4: estimate water year peff/precipitation fraction (for model check)
# Step 5: sum scaled monthly peff to growing season (with added 3 months' peff before growing season to consider carried over soil moisture storage)
# Step 6: sum scaled monthly peff to growing season (without considering carried over soil moisture)

if __name__ == '__main__':
    monthly_model_version = 'v19'                   #####
    water_yr_model_version = 'v20'                  #####
    skip_estimating_peff_water_yr_total = False     #####
    skip_peff_monthly_scaling = False               #####
    skip_sum_scaled_peff_water_year = False         #####
    skip_peff_frac_estimate_water_yr = False        #####
    skip_sum_scale_peff_to_gs_with_SM = False       #####
    skip_sum_scale_peff_to_gs = False               #####

    # # # # # Step 1: water year peff raster creation using water year peff fraction # # # # #
    years = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
             2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
             2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
             2024)
    water_year_precip_dir = '../../Data_main/Raster_data/GRIDMET_Precip/WestUS_water_year/sum'
    water_year_peff_frac_dir = f'../../Data_main/Raster_data/Effective_precip_fraction_WestUS/{water_yr_model_version}_water_year_frac'
    output_updated_peff_water_yr_dir = f'../../Data_main/Raster_data/Effective_precip_fraction_WestUS/{water_yr_model_version}_water_year_total_from_fraction'

    estimate_water_yr_peff_using_peff_frac(years, water_year_precip_dir, water_year_peff_frac_dir,
                                           output_updated_peff_water_yr_dir,
                                           skip_processing=skip_estimating_peff_water_yr_total)

    # # # # #  Step 2: scaling monthly peff prediction with annual model # # # # #
    years = (1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
             2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
             2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
             2024)
    unscaled_peff_monthly_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_monthly'
    unscaled_peff_water_yr_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_water_year'
    peff_monthly_scaled_output_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_monthly_scaled'

    scale_monthy_peff_with_wateryr_peff_model(years, unscaled_peff_monthly_dir, unscaled_peff_water_yr_dir,
                                              output_updated_peff_water_yr_dir, peff_monthly_scaled_output_dir,
                                              skip_processing=skip_peff_monthly_scaling)

    # # # # # Step 3: summing scaled monthly effective precipitation for water year # # # # #
    output_peff_scaled_water_year_summed_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_water_year_scaled'

    sum_cropET_water_yr(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006,
                                    2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                                    2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
                                    2023, 2024),
                        input_cropET_monthly_dir=peff_monthly_scaled_output_dir,
                        output_dir_water_yr=output_peff_scaled_water_year_summed_dir,
                        save_keyword='effective_precip',
                        skip_processing=skip_sum_scaled_peff_water_year)

    # # # # #  Step4: estimating water year peff fraction for scaled peff data # # # # #
    scaled_frac_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_peff_fraction_scaled'

    estimate_peff_precip_water_year_fraction(years_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                                         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                                         2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
                                                         2024),
                                             peff_dir_water_yr=output_peff_scaled_water_year_summed_dir,
                                             precip_dir_water_yr=water_year_precip_dir,
                                             output_dir=scaled_frac_dir,
                                             skip_processing=skip_peff_frac_estimate_water_yr)

    # # # # #  Step 5: compile scaled monthly Peff to growing season including 3 months lagged Peff as soil moisture storage # # # # #
    output_peff_grow_season_summed_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_grow_season_scaled_with_SM'

    dynamic_gs_sum_peff_with_3m_SM_storage(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                                      2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                                      2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023),
                                           growing_season_dir='../../Data_main/Raster_data/Growing_season',
                                           monthly_input_dir=peff_monthly_scaled_output_dir,
                                           gs_output_dir=output_peff_grow_season_summed_dir,
                                           skip_processing=skip_sum_scale_peff_to_gs_with_SM)

    # # # # #  Step 6: compile scaled monthly Peff to growing season (without considering additional soil mositure storage from previous months) # # # # #
    final_peff_grow_season_summed_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{monthly_model_version}_grow_season_scaled'

    dynamic_gs_sum_ET(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                                 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023),
                      growing_season_dir='../../Data_main/Raster_data/Growing_season',
                      monthly_input_dir=peff_monthly_scaled_output_dir,
                      gs_output_dir=final_peff_grow_season_summed_dir,
                      sum_keyword='effective_precip',
                      skip_processing=skip_sum_scale_peff_to_gs)
