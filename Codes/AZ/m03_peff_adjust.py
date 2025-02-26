import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.AZ.az_utils import scale_monthy_peff_with_wateryr_peff_model
from Codes.data_download_preprocess.preprocesses import sum_cropET_water_yr, dynamic_gs_sum_peff_with_3m_SM_storage, \
    dynamic_gs_sum_ET
from Codes.effective_precip.m00_eff_precip_utils import estimate_peff_precip_water_year_fraction, \
    estimate_water_yr_peff_using_peff_frac


# # # Steps

# Step 1: water year peff raster creation using water year peff fraction (water year precip * water year fraction)
# Step 2: scaling monthly peff prediction with water year model
# Step 3: sum scaled monthly peff to growing season (with added 3 months' peff before growing season to consider carried over soil moisture storage)
# Step 4: sum scaled monthly peff to growing season (without considering carried over soil moisture)

if __name__ == '__main__':
    monthly_model_version = 'v19'                    #####
    water_yr_model_version = 'v20'                   #####
    skip_estimating_peff_water_yr_total = False      #####
    skip_peff_monthly_scaling = False                #####
    skip_sum_scaled_peff_water_year = False          #####
    skip_peff_frac_estimate_water_yr = False         #####
    skip_sum_scale_peff_to_gs_with_SM = False        #####
    skip_sum_scale_peff_to_gs = False                #####

    # # # # # Step 1: water year peff raster creation using water year peff fraction # # # # #
    years = list(range(1986, 2021))
    water_year_precip_dir = '../../Data_main/AZ_files/rasters/GRIDMET_Precip/WestUS_water_year/sum'
    water_year_peff_frac_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_fraction_WestUS/{water_yr_model_version}_water_year_frac'
    output_updated_peff_water_yr_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_fraction_WestUS/{water_yr_model_version}_water_year_total_from_fraction'

    estimate_water_yr_peff_using_peff_frac(years, water_year_precip_dir, water_year_peff_frac_dir,
                                           output_updated_peff_water_yr_dir,
                                           skip_processing=skip_estimating_peff_water_yr_total)

    # # # # #  Step 2: scaling monthly peff prediction with annual model # # # # #
    years = list(range(1985, 2021))         # the Peff data will be scaled from month 10 of 1985 to month 9 of 2020

    unscaled_peff_monthly_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{monthly_model_version}_monthly'
    unscaled_peff_water_yr_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{monthly_model_version}_water_year'
    peff_monthly_scaled_output_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{monthly_model_version}_monthly_scaled'

    scale_monthy_peff_with_wateryr_peff_model(years, unscaled_peff_monthly_dir, unscaled_peff_water_yr_dir,
                                              output_updated_peff_water_yr_dir, peff_monthly_scaled_output_dir,
                                              skip_processing=skip_peff_monthly_scaling)


    # # # # #  Step 5: compile scaled monthly Peff to growing season including 3 months lagged Peff as soil moisture storage # # # # #
    output_peff_grow_season_summed_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{monthly_model_version}_grow_season_scaled_with_SM'

    dynamic_gs_sum_peff_with_3m_SM_storage(year_list=list(range(1986, 2020)),  # can't do 2020 as month 10-12's data wasn't scaled
                                           growing_season_dir='../../Data_main/AZ_files/rasters/Growing_season',
                                           monthly_input_dir=peff_monthly_scaled_output_dir,
                                           gs_output_dir=output_peff_grow_season_summed_dir,
                                           skip_processing=skip_sum_scale_peff_to_gs_with_SM)

    # # # # #  Step 6: compile scaled monthly Peff to growing season (without considering additional soil mositure storage from previous months) # # # # #
    final_peff_grow_season_summed_dir = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{monthly_model_version}_grow_season_scaled'

    dynamic_gs_sum_ET(year_list=list(range(1986, 2020)),  # can't do 2020 as month 10-12's data wasn't scaled
                      growing_season_dir='../../Data_main/AZ_files/rasters/Growing_season',
                      monthly_input_dir=peff_monthly_scaled_output_dir,
                      gs_output_dir=final_peff_grow_season_summed_dir,
                      sum_keyword='effective_precip',
                      skip_processing=skip_sum_scale_peff_to_gs)