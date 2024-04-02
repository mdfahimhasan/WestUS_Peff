import sys

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.raster_ops import shapefile_to_raster
from Codes.results_analysis.analysis_utils import run_annual_csv_processing, run_annual_csv_processing_DV, \
    compile_annual_pumping_netGW_all_basins

model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
basin_name_dict = {'GMD4': 'GMD4, KS', 'GMD3': 'GMD3, KS', 'RPB': 'RPB, CO',
                   'Harquahala': 'Harquahala INA, AZ', 'Douglas': 'Douglas AMA, AZ',
                   'Diamond': 'Diamond Valley, NV', 'Harney': 'Harney Basin, Or'}

if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # boolean switches to process data for each region
    skip_process_gmd4_data = True   # Groundwater Management District 4, KS
    skip_process_gmd3_data = True  # Groundwater Management District 3, KS
    skip_process_rpb_data = True    # Republican River Basin, Co
    skip_process_hqr_data = True   # Harquahala INA, AZ
    skip_process_doug_data = True  # Douglas AMA, AZ
    skip_process_dv_data = True   # Diamond valley, NV
    skip_process_hrn_data = False   # Harney Basin, OR
    skip_compile_AF_annual_data = True  # all basins

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 4 (GMD4), Kansas
    if not skip_process_gmd4_data:
        print('Processing netGW, pumping (in-situ + USGS) annual dataframe for GMD4, KS...')

        years = [2016, 2017, 2018, 2019, 2020]
        basin_code = 'gmd4'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
        westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        westUS_irr_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        pumping_pts_shp = '../../Data_main/Pumping/Kansas/KS_GW_GMD4.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        main_output_dir = f'../../Data_main/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        usgs_westUS_GW_shp = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_GW.shp'
        usgs_annual_GW_estimates_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_USGS.csv'
        final_annual_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing(years, basin_code, basin_shp,
                                  westUS_netGW_dir, westUS_irr_frac_dir,
                                  pumping_pts_shp, pumping_attr_AF, year_attr,
                                  main_output_dir, pixelwise_output_csv,
                                  usgs_westUS_GW_shp,
                                  usgs_annual_GW_estimates_csv,
                                  final_annual_csv, skip_processing=skip_process_gmd4_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 3 (GMD3), Kansas
    if not skip_process_gmd3_data:
        print('Processing netGW, pumping (in-situ + USGS) annual dataframe for GMD3, KS...')

        years = [2016, 2017, 2018, 2019, 2020]
        basin_code = 'gmd3'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
        westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        westUS_irr_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        pumping_pts_shp = '../../Data_main/Pumping/Kansas/KS_GW_GMD3.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        main_output_dir = f'../../Data_main/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        usgs_westUS_GW_shp = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_GW.shp'
        usgs_annual_GW_estimates_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_USGS.csv'
        final_annual_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing(years, basin_code, basin_shp,
                                  westUS_netGW_dir, westUS_irr_frac_dir,
                                  pumping_pts_shp, pumping_attr_AF, year_attr,
                                  main_output_dir, pixelwise_output_csv,
                                  usgs_westUS_GW_shp,
                                  usgs_annual_GW_estimates_csv,
                                  final_annual_csv, skip_processing=skip_process_gmd3_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Republican River Basin (RPB), Colorado
    if not skip_process_rpb_data:
        print('Processing netGW, pumping (in-situ + USGS) annual dataframe for Republican Basin, CO...')

        years = [2016, 2017, 2018, 2019, 2020]
        basin_code = 'rpb'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
        westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        westUS_irr_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        pumping_pts_shp = '../../Data_main/Pumping/Colorado/CO_GW_RPB.shp'
        pumping_attr_AF = 'ann_amt'
        year_attr = 'irr_year'
        main_output_dir = f'../../Data_main/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        usgs_westUS_GW_shp = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_GW.shp'
        usgs_annual_GW_estimates_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_USGS.csv'
        final_annual_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing(years, basin_code, basin_shp,
                                  westUS_netGW_dir, westUS_irr_frac_dir,
                                  pumping_pts_shp, pumping_attr_AF, year_attr,
                                  main_output_dir, pixelwise_output_csv,
                                  usgs_westUS_GW_shp,
                                  usgs_annual_GW_estimates_csv,
                                  final_annual_csv, skip_processing=skip_process_rpb_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Harquahala INA, Arizona
    if not skip_process_hqr_data:
        print('Processing netGW, pumping (in-situ + USGS) annual dataframe for Harquahala INA, AZ...')

        years = [2016, 2017, 2018, 2019, 2020]
        basin_code = 'hqr'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp'
        westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        westUS_irr_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        pumping_pts_shp = '../../Data_main/Pumping/Arizona/AZ_GW_Harquahala.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        main_output_dir = f'../../Data_main/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        usgs_westUS_GW_shp = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_GW.shp'
        usgs_annual_GW_estimates_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_USGS.csv'
        final_annual_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing(years, basin_code, basin_shp,
                                  westUS_netGW_dir, westUS_irr_frac_dir,
                                  pumping_pts_shp, pumping_attr_AF, year_attr,
                                  main_output_dir, pixelwise_output_csv,
                                  usgs_westUS_GW_shp,
                                  usgs_annual_GW_estimates_csv,
                                  final_annual_csv, skip_processing=skip_process_hqr_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Douglas AMA, Arizona
    if not skip_process_doug_data:
        print('Processing netGW, pumping (in-situ + USGS) dataset and netGW-pumping dataframe for Douglas AMA, AZ...')

        years = [2016, 2017, 2018, 2019, 2020]
        basin_code = 'doug'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp'
        westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        westUS_irr_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        pumping_pts_shp = '../../Data_main/Pumping/Arizona/AZ_GW_Douglas.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        main_output_dir = f'../../Data_main/results_eval/netGW/{basin_code}'
        pixelwise_output_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_netGW_pumping.csv'
        usgs_westUS_GW_shp = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_GW.shp'
        usgs_annual_GW_estimates_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_USGS.csv'
        final_annual_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing(years, basin_code, basin_shp,
                                  westUS_netGW_dir, westUS_irr_frac_dir,
                                  pumping_pts_shp, pumping_attr_AF, year_attr,
                                  main_output_dir, pixelwise_output_csv,
                                  usgs_westUS_GW_shp,
                                  usgs_annual_GW_estimates_csv,
                                  final_annual_csv, skip_processing=skip_process_doug_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Diamond Valley, Nevada
    if not skip_process_dv_data:
        print('Processing netGW, pumping (in-situ + USGS) dataset and netGW-pumping dataframe for Diamond Valley, NV...')

        years = [2016, 2017, 2018, 2019, 2020]
        basin_code = 'dv'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp'
        westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        pumping_csv = '../../Data_main/Pumping/Nevada/joined_data/dv_joined_et_pumping_data_all.csv'
        pumping_attr_AF = 'pumping_AF'
        main_output_dir = f'../../Data_main/results_eval/netGW/{basin_code}'
        usgs_westUS_GW_shp = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_GW.shp'
        usgs_annual_GW_estimates_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_USGS.csv'
        final_annual_csv = f'../../Data_main/results_eval/netGW/{basin_code}/{basin_code}_annual.csv'

        run_annual_csv_processing_DV(years, basin_code, basin_shp,
                                     westUS_netGW_dir,
                                     pumping_csv, pumping_attr_AF,
                                     main_output_dir,
                                     usgs_westUS_GW_shp,
                                     usgs_annual_GW_estimates_csv,
                                     final_annual_csv,
                                     skip_processing=skip_process_dv_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For all basins' annual csv compilation
    if not skip_compile_AF_annual_data:
        print('Compiling annual csv of all basins...')

        basin_annual_csvs = ['../../Data_main/results_eval/netGW/gmd4/gmd4_annual.csv',
                             '../../Data_main/results_eval/netGW/gmd3/gmd3_annual.csv',
                             '../../Data_main/results_eval/netGW/rpb/rpb_annual.csv',
                             '../../Data_main/results_eval/netGW/hqr/hqr_annual.csv',
                             '../../Data_main/results_eval/netGW/doug/doug_annual.csv',
                             '../../Data_main/results_eval/netGW/dv/dv_annual.csv']
        output_csv = '../../Data_main/results_eval/netGW/annual_all_basins.csv'

        compile_annual_pumping_netGW_all_basins(annual_csv_list=basin_annual_csvs, output_csv=output_csv)
