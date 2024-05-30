import sys

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.results_analysis.analysis_utils import run_growS_peff_water_yr_precip_compilation, \
    run_monthly_peff_precip_compilation, compile_peff_precip_csv_all_basins

model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

model_version = 'v10'  # # # #

# # # # # for growing season # # # # #
if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # boolean switches to process data for each region
    skip_process_gmd4_data = True  # Groundwater Management District 4, KS
    skip_process_gmd3_data = True  # Groundwater Management District 3, KS
    skip_process_rpb_data = True  # Republican River Basin, Co
    skip_process_hqr_data = True  # Harquahala INA, AZ
    skip_process_doug_data = True  # Douglas AMA, AZ
    skip_process_dv_data = True  # Diamond Valley, NV
    skip_process_cv_data = True  # Central Valley, CA
    skip_compile_all_basin_data = True  # all basins

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 4 (GMD4), Kansas
    if not skip_process_gmd4_data:
        print('Processing growing season effective precipitation and water year precipitation for GMD4, KS...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'gmd4'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_gmd4_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 3 (GMD3), Kansas
    if not skip_process_gmd3_data:
        print('Processing growing season effective precipitation and water year precipitation for GMD3, KS...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'gmd3'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_gmd3_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Republican River Basin (RPB), Colorado
    if not skip_process_rpb_data:
        print(
            'Processing growing season effective precipitation and water year precipitation for Republican Basin, CO...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'rpb'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_rpb_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Harquahala INA, Arizona
    if not skip_process_hqr_data:
        print(
            'Processing growing season effective precipitation and water year precipitation for Harquahala INA, AZ...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'hqr'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_hqr_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Douglas AMA, Arizona
    if not skip_process_doug_data:
        print('Processing growing season effective precipitation and water year precipitation for Douglas AMA, AZ...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'doug'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_doug_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Diamond Valley, Nevada
    if not skip_process_dv_data:
        print(
            'Processing growing season effective precipitation and water year precipitation for Diamond Valley, NV...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'dv'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_dv_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Central Valley, California
    if not skip_process_cv_data:
        print(
            'Processing growing season effective precipitation and water year precipitation for Central Valley, CA...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'cv'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Central_Valley.shp'
        growS_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
        water_yr_precip_dir = '../../Data_main/Raster_data/PRISM_Precip/WestUS_water_year'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/grow_season/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/water_yr/{basin_code}'
        output_csv = f'../../Data_main/results_eval/peff/grow_season/{basin_code}.csv'

        run_growS_peff_water_yr_precip_compilation(years=years, basin_shp=basin_shp,
                                                   Peff_input_dir=growS_peff_dir,
                                                   basin_Peff_output_dir=basin_Peff_output_dir,
                                                   precip_input_dir=water_yr_precip_dir,
                                                   basin_precip_output_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=output_csv,
                                                   skip_processing=skip_process_cv_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For all basins' annual csv compilation
    if not skip_compile_all_basin_data:
        print('Compiling csv of all basins...')

        basin_csvs = ['../../Data_main/results_eval/peff/grow_season/gmd4.csv',
                      '../../Data_main/results_eval/peff/grow_season/gmd3.csv',
                      '../../Data_main/results_eval/peff/grow_season/rpb.csv',
                      '../../Data_main/results_eval/peff/grow_season/hqr.csv',
                      '../../Data_main/results_eval/peff/grow_season/doug.csv',
                      '../../Data_main/results_eval/peff/grow_season/dv.csv',
                      '../../Data_main/results_eval/peff/grow_season/cv.csv']
        output_csv = '../../Data_main/results_eval/peff/grow_season/all_basins.csv'

        compile_peff_precip_csv_all_basins(annual_csv_list=basin_csvs, output_csv=output_csv)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # for months # # # # #
if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # boolean switches to process data for each region
    skip_process_gmd4_data = False  # Groundwater Management District 4, KS
    skip_process_gmd3_data = False  # Groundwater Management District 3, KS
    skip_process_rpb_data = False  # Republican River Basin, Co
    skip_process_hqr_data = False  # Harquahala INA, AZ
    skip_process_doug_data = False  # Douglas AMA, AZ
    skip_process_dv_data = False  # Diamond Valley, NV
    skip_process_cv_data = False  # Central Valley, CA
    skip_compile_all_basin_peff_data = False  # all basins
    skip_compile_all_basin_mv_avg_precip_data = False  # all basins

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 4 (GMD4), Kansas
    if not skip_process_gmd4_data:
        print('Processing monthly effective precipitation for GMD4, KS...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'gmd4'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_gmd4_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 3 (GMD3), Kansas
    if not skip_process_gmd3_data:
        print('Processing monthly effective precipitation for GMD3, KS...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'gmd3'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_gmd3_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Republican River Basin (RPB), Colorado
    if not skip_process_rpb_data:
        print(
            'Processing monthly effective precipitation for Republican Basin, CO...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'rpb'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_rpb_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Harquahala INA, Arizona
    if not skip_process_hqr_data:
        print(
            'Processing monthly effective precipitation for Harquahala INA, AZ...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'hqr'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_hqr_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Douglas AMA, Arizona
    if not skip_process_doug_data:
        print('Processing monthly effective precipitation for Douglas AMA, AZ...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'doug'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_doug_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Diamond Valley, Nevada
    if not skip_process_dv_data:
        print(
            'Processing monthly effective precipitation for Diamond Valley, NV...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'dv'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_dv_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Central Valley, California
    if not skip_process_cv_data:
        print(
            'Processing monthly effective precipitation for Central Valley, CA...')

        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        basin_code = 'cv'
        basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Central_Valley.shp'
        monthly_peff_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly'
        monthly_precip_dir = f'../../Data_main/Raster_data/PRISM_Precip/WestUS_monthly'
        basin_Peff_output_dir = f'../../Data_main/results_eval/peff/monthly/{basin_code}'
        basin_precip_output_dir = f'../../Data_main/results_eval/precip/monthly/{basin_code}'
        peff_output_csv = f'../../Data_main/results_eval/peff/monthly/{basin_code}.csv'
        precip_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}.csv'
        precip_moving_avg_output_csv = f'../../Data_main/results_eval/precip/monthly/{basin_code}_mv_avg.csv'

        run_monthly_peff_precip_compilation(years=years,
                                            basin_shp=basin_shp,
                                            Peff_input_dir=monthly_peff_dir,
                                            precip_input_dir=monthly_precip_dir,
                                            basin_Peff_output_dir=basin_Peff_output_dir,
                                            basin_precip_output_dir=basin_precip_output_dir,
                                            basin_code=basin_code, peff_output_csv=peff_output_csv,
                                            precip_output_csv=precip_output_csv,
                                            precip_moving_avg_output_csv=precip_moving_avg_output_csv,
                                            skip_processing=skip_process_cv_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For all basins' effective precip csv compilation
    if not skip_compile_all_basin_peff_data:
        print('Compiling csv of all basins...')

        basin_csvs = ['../../Data_main/results_eval/peff/monthly/gmd4.csv',
                      '../../Data_main/results_eval/peff/monthly/gmd3.csv',
                      '../../Data_main/results_eval/peff/monthly/rpb.csv',
                      '../../Data_main/results_eval/peff/monthly/hqr.csv',
                      '../../Data_main/results_eval/peff/monthly/doug.csv',
                      '../../Data_main/results_eval/peff/monthly/dv.csv',
                      '../../Data_main/results_eval/peff/monthly/cv.csv']
        output_csv = '../../Data_main/results_eval/peff/monthly/all_basins.csv'

        compile_peff_precip_csv_all_basins(annual_csv_list=basin_csvs, output_csv=output_csv)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For all basins' moving avg monthly precip csv compilation
    if not skip_compile_all_basin_mv_avg_precip_data:
        print('Compiling csv of all basins...')

        basin_csvs = ['../../Data_main/results_eval/precip/monthly/gmd4_mv_avg.csv',
                      '../../Data_main/results_eval/precip/monthly/gmd3_mv_avg.csv',
                      '../../Data_main/results_eval/precip/monthly/rpb_mv_avg.csv',
                      '../../Data_main/results_eval/precip/monthly/hqr_mv_avg.csv',
                      '../../Data_main/results_eval/precip/monthly/doug_mv_avg.csv',
                      '../../Data_main/results_eval/precip/monthly/dv_mv_avg.csv',
                      '../../Data_main/results_eval/precip/monthly/cv_mv_avg.csv']
        output_csv = '../../Data_main/results_eval/precip/monthly/all_basins_mv_avg.csv'

        compile_peff_precip_csv_all_basins(annual_csv_list=basin_csvs, output_csv=output_csv)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
