from Codes.netGW.netGW_Irr import estimate_netGW_Irr
from Codes.AZ.az_utils import run_annual_csv_processing_AZ

AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'

if __name__ == '__main__':
    # flags
    skip_AZ_netGW_processing = True         ######
    skip_process_hqr_data = False           ###### Harquahala INA, AZ
    skip_process_doug_data = False          ###### Douglas AMA, AZ


    # # # estimating netGW for Arizona
    years = list(range(2000, 2020))            # limited from 2000-2019 due to SW_Irr data unavailability

    model_version = 'v19'
    effective_precip = f'../../Data_main/AZ_files/rasters/Effective_precip_prediction_WestUS/{model_version}_grow_season_scaled_with_SM'
    irrigated_cropET = '../../Data_main/AZ_files/rasters/Irrigated_cropET/WestUS_grow_season'
    irrigated_fraction = '../../Data_main/AZ_files/rasters/Irrigated_cropland/Irrigated_Frac'
    sw_irrigation_dir = '../../Data_main/AZ_files/rasters/SW_irrigation'
    netGW_irrigation_output_dir = '../../Data_main/AZ_files/rasters/NetGW_irrigation/WesternUS'

    estimate_netGW_Irr(years_list=years, effective_precip_dir_pp=effective_precip,
                       irrigated_cropET_dir=irrigated_cropET,
                       irrigated_fraction_dir=irrigated_fraction, sw_cnsmp_use_dir=sw_irrigation_dir,
                       output_dir=netGW_irrigation_output_dir, ref_raster=AZ_raster,
                       skip_processing=skip_AZ_netGW_processing)

    # # # For Harquahala INA, Arizona
    if not skip_process_hqr_data:
        print('Processing netGW, pumping (in-situ + USGS) annual dataframe for Harquahala INA, AZ...')

        years = list(range(2000, 2020))
        basin_code = 'hqr'
        basin_shp = '../../Data_main/AZ_files/ref_files/Harquahala_INA.shp'
        az_netGW_dir = '../../Data_main/AZ_files/rasters/NetGW_irrigation/WesternUS'
        annual_pumping_csv = f'../../Data_main/Pumping/Arizona/pumping_AZ_v2.csv'
        main_output_dir = f'../../Data_main/AZ_files/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/AZ_files/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        final_annual_csv = f'../../Data_main/AZ_files/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing_AZ(years, basin_code, basin_shp,
                                     az_netGW_dir, annual_pumping_csv,
                                     main_output_dir, pixelwise_output_csv,
                                     final_annual_csv,
                                     skip_processing=skip_process_hqr_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Douglas AMA, Arizona
    if not skip_process_doug_data:
        print('Processing netGW, pumping (in-situ + USGS) dataset and netGW-pumping dataframe for Douglas AMA, AZ...')

        years = list(range(2000, 2020))
        basin_code = 'doug'
        basin_shp = '../../Data_main/AZ_files/ref_files/Douglas_AMA.shp'
        az_netGW_dir = '../../Data_main/AZ_files/rasters/NetGW_irrigation/WesternUS'
        annual_pumping_csv = f'../../Data_main/Pumping/Arizona/pumping_AZ_v2.csv'
        main_output_dir = f'../../Data_main/AZ_files/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/AZ_files/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        final_annual_csv = f'../../Data_main/AZ_files/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing_AZ(years, basin_code, basin_shp,
                                     az_netGW_dir, annual_pumping_csv,
                                     main_output_dir, pixelwise_output_csv,
                                     final_annual_csv,
                                     skip_processing=skip_process_doug_data)