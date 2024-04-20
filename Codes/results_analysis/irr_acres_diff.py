from Codes.results_analysis.analysis_utils import process_and_aggregate_irrigated_acres, compile_irr_acres_all_basins

model_res = 0.01976293625031605786  # in deg, ~2 km

if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # boolean switches to process data for each region
    skip_process_gmd4_data = False   # Groundwater Management District 4, KS
    skip_process_gmd3_data = False  # Groundwater Management District 3, KS
    skip_process_rpb_data = False  # Republican River Basin, CO
    skip_process_dv_data = False  # Diamond Valley, NV
    skip_compile_all_basin_irr_acreage = False  # all basins

    # basic input data folder
    years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
             2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    irr_cropland_dir = '../../Data_main/Raster_data/Irrigated_cropland'
    irr_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 4 (GMD4), Kansas
    if not skip_process_gmd4_data:
        basin_name = 'gmd4'
        gmd4_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
        ks_pumping_acres_csv = '../../Data_main/Pumping/Kansas/csv/pumping_KS_acres_2000_2020.csv'
        output_dir = f'../../Data_main/results_eval/irr_acres/{basin_name}'
        output_csv = f'../../Data_main/results_eval/irr_acres/{basin_name}/irr_acres.csv'

        process_and_aggregate_irrigated_acres(years=years,
                                              irr_cropland_input_dir=irr_cropland_dir,
                                              irr_frac_input_dir=irr_frac_dir,
                                              basin_shp=gmd4_shp,
                                              basin_name=basin_name,
                                              in_situ_areacsv=ks_pumping_acres_csv,
                                              main_output_dir=output_dir,
                                              output_csv=output_csv,
                                              resolution=model_res,
                                              in_situ_shp_list_CO=None)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 3 (GMD3), Kansas
    if not skip_process_gmd3_data:
        basin_name = 'gmd3'
        gmd3_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
        ks_pumping_acres_csv = '../../Data_main/Pumping/Kansas/csv/pumping_KS_acres_2000_2020.csv'
        output_dir = f'../../Data_main/results_eval/irr_acres/{basin_name}'
        output_csv = f'../../Data_main/results_eval/irr_acres/{basin_name}/irr_acres.csv'

        process_and_aggregate_irrigated_acres(years=years,
                                              irr_cropland_input_dir=irr_cropland_dir,
                                              irr_frac_input_dir=irr_frac_dir,
                                              basin_shp=gmd3_shp,
                                              basin_name=basin_name,
                                              in_situ_areacsv=ks_pumping_acres_csv,
                                              main_output_dir=output_dir,
                                              output_csv=output_csv,
                                              resolution=model_res,
                                              in_situ_shp_list_CO=None)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Republican RIver Basin, CO
    if not skip_process_rpb_data:
        basin_name = 'rpb'
        rpb_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
        output_dir = f'../../Data_main/results_eval/irr_acres/{basin_name}'
        output_csv = f'../../Data_main/results_eval/irr_acres/{basin_name}/irr_acres.csv'
        irr_field_list = ['../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2010.shp',
                          '../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2015.shp',
                          '../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2016.shp',
                          '../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2017.shp',
                          '../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2018.shp',
                          '../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2019.shp',
                          '../../Data_main/Pumping/Colorado/Republican_Shapefiles/Repub_Irrig_2020.shp']

        process_and_aggregate_irrigated_acres(years=years, irr_cropland_input_dir=irr_cropland_dir,
                                              irr_frac_input_dir=irr_frac_dir,
                                              basin_shp=rpb_shp, basin_name=basin_name,
                                              main_output_dir=output_dir, output_csv=output_csv,
                                              resolution=model_res,
                                              in_situ_shp_list_CO=irr_field_list,
                                              in_situ_areacsv=None)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Diamond Valley, Nv
    if not skip_process_dv_data:
        basin_name = 'dv'
        dv_shp = '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp'
        pumping_area_csv = '../../Data_main/Pumping/Nevada/joined_data/dv_joined_et_pumping_data_all.csv'
        output_dir = f'../../Data_main/results_eval/irr_acres/{basin_name}'
        output_csv = f'../../Data_main/results_eval/irr_acres/{basin_name}/irr_acres.csv'

        process_and_aggregate_irrigated_acres(years=years,
                                              irr_cropland_input_dir=irr_cropland_dir,
                                              irr_frac_input_dir=irr_frac_dir,
                                              basin_shp=dv_shp,
                                              basin_name=basin_name,
                                              in_situ_areacsv=pumping_area_csv,
                                              main_output_dir=output_dir,
                                              output_csv=output_csv,
                                              resolution=model_res,
                                              in_situ_shp_list_CO=None)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For all basins' annual csv compilation
    if not skip_compile_all_basin_irr_acreage:
        print('Compiling annual irrigation acreage csv of all basins...')

        basin_annual_csvs = ['../../Data_main/results_eval/irr_acres/gmd4/irr_acres.csv',
                             '../../Data_main/results_eval/irr_acres/gmd3/irr_acres.csv',
                             '../../Data_main/results_eval/irr_acres/rpb/irr_acres.csv',
                             '../../Data_main/results_eval/irr_acres/dv/irr_acres.csv']
        output_csv = '../../Data_main/results_eval/irr_acres/all_irr_acres.csv'

        compile_irr_acres_all_basins(annual_csv_list=basin_annual_csvs, output_csv=output_csv)
