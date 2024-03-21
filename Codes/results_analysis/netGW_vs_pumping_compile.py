import sys

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.netGW.netGW_utils import clip_netGW_Irr_frac_for_basin, pumping_AF_pts_to_mm_raster, \
    compile_basin_df_for_netGW_pumping, extract_pumping_estimate_with_lat_lon, \
    aggregate_pixelCSV_to_annualCSV, aggregate_netGW_pumping_to_annual,\
    compile_annual_AF_pumping_netGW_all_basins

model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # boolean switches to process data for each region
    skip_process_gmd4_data = True
    skip_process_gmd3_data = True
    skip_process_rpb_data = True
    skip_process_hqr_data = True
    skip_process_doug_data = True
    skip_process_dv_data = False
    skip_compile_AF_annual_data = True

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 4 (GMD4), Kansas
    if not skip_process_gmd4_data:
        print('Processing netGW, pumping dataset and netGW-pumping dataframe for GMD4, KS...')

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for GMD4, KS
        print('# # # # #  STEP 1 # # # # #')

        years = [2016, 2017, 2018, 2019, 2020]
        GMD4_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
        netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/netGW_GMD4_KS'
        irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        basin_irrig_fraction_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/irrig_frac'

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=GMD4_shp,
                                      netGW_input_dir=netGW_irrigation_dir,
                                      basin_netGW_output_dir=basin_netGW_output_dir,
                                      irr_frac_input_dir=irrig_fraction_dir,
                                      basin_irr_frac_output_dir=basin_irrig_fraction_output_dir,
                                      resolution=model_res, skip_processing=skip_process_gmd4_data)

        # # # # #  STEP 2 # # # # #
        # # Converting annual pumping shapefile (unit AF) to mm raster
        print('# # # # #  STEP 2 # # # # #')

        pumping_shp = '../../Data_main/Pumping/Kansas/KS_GW_GMD4.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        GMD4_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS'

        pumping_AF_pts_to_mm_raster(years=years, irrigated_fraction_dir=irrig_fraction_dir,
                                    pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                                    year_attr=year_attr,
                                    output_dir=GMD4_output_dir, basin_shp=GMD4_shp, ref_raster=WestUS_raster,
                                    resolution=model_res, skip_processing=skip_process_gmd4_data)

        # # # # #  STEP 3 # # # # #
        # # Compile growing season netGW and annual pumping in dataframes
        print('# # # # #  STEP 3 # # # # #')

        basin_pumping_mm_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/pumping_mm'
        basin_pumping_AF_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/pumping_AF_raster'
        GMD4_csv = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/KS_GMD4_netGW_pumping.csv'

        compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_output_dir,
                                           basin_pumping_mm_dir=basin_pumping_mm_dir,
                                           basin_pumping_AF_dir=basin_pumping_AF_dir,
                                           output_csv=GMD4_csv, skip_processing=skip_process_gmd4_data)

        annual_csv = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/KS_GMD4_annual.csv'
        aggregate_pixelCSV_to_annualCSV(pixel_csv=GMD4_csv, output_annual_csv=annual_csv)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Groundwater Management District 3 (GMD3), Kansas
    if not skip_process_gmd3_data:
        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for GMD3, KS
        print('Processing netGW, pumping dataset and netGW-pumping dataframe for GMD3, KS...')
        print('# # # # #  STEP 1 # # # # #')

        years = [2016, 2017, 2018, 2019, 2020]
        GMD3_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
        netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/netGW_GMD3_KS'
        irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        basin_irrig_fraction_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/irrig_frac'

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=GMD3_shp,
                                      netGW_input_dir=netGW_irrigation_dir,
                                      basin_netGW_output_dir=basin_netGW_output_dir,
                                      irr_frac_input_dir=irrig_fraction_dir,
                                      basin_irr_frac_output_dir=basin_irrig_fraction_output_dir,
                                      resolution=model_res, skip_processing=skip_process_gmd3_data)

        # # # # #  STEP 2 # # # # #
        # # Converting annual pumping shapefile (unit AF) to mm raster
        print('# # # # #  STEP 2 # # # # #')

        pumping_shp = '../../Data_main/Pumping/Kansas/KS_GW_GMD3.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        GMD3_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS'

        pumping_AF_pts_to_mm_raster(years=years, irrigated_fraction_dir=irrig_fraction_dir,
                                    pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                                    year_attr=year_attr, output_dir=GMD3_output_dir, basin_shp=GMD3_shp,
                                    ref_raster=WestUS_raster, resolution=model_res,
                                    skip_processing=skip_process_gmd3_data)

        # # # # #  STEP 3 # # # # #
        # # Compile growing season netGW and annual pumping in dataframes
        print('# # # # #  STEP 3 # # # # #')

        basin_pumping_mm_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/pumping_mm'
        basin_pumping_AF_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/pumping_AF_raster'
        GMD3_csv = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/KS_GMD3_netGW_pumping.csv'

        compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_output_dir,
                                           basin_pumping_mm_dir=basin_pumping_mm_dir,
                                           basin_pumping_AF_dir=basin_pumping_AF_dir,
                                           output_csv=GMD3_csv, skip_processing=skip_process_gmd3_data)

        annual_csv = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/KS_GMD3_annual.csv'
        aggregate_pixelCSV_to_annualCSV(pixel_csv=GMD3_csv, output_annual_csv=annual_csv)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Republican River Basin (RPB), Colorado
    if not skip_process_rpb_data:
        print('Processing netGW, pumping dataset and netGW-pumping dataframe for RPB, CO...')

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for RPB, CO
        print('# # # # #  STEP 1 # # # # #')

        years = [2016, 2017, 2018, 2019, 2020]
        RPB_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
        netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/netGW_RPB_CO'
        irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        basin_irrig_fraction_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/irrig_frac'

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=RPB_shp,
                                      netGW_input_dir=netGW_irrigation_dir,
                                      basin_netGW_output_dir=basin_netGW_output_dir,
                                      irr_frac_input_dir=irrig_fraction_dir,
                                      basin_irr_frac_output_dir=basin_irrig_fraction_output_dir,
                                      resolution=model_res, skip_processing=skip_process_rpb_data)

        # # # # #  STEP 2 # # # # #
        # # Converting annual pumping shapefile (unit AF) to mm raster
        print('# # # # #  STEP 2 # # # # #')

        pumping_shp = '../../Data_main/Pumping/Colorado/CO_GW_RPB.shp'
        pumping_attr_AF = 'ann_amt'
        year_attr = 'irr_year'
        RPB_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO'

        pumping_AF_pts_to_mm_raster(years=years, irrigated_fraction_dir=irrig_fraction_dir,
                                    pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                                    year_attr=year_attr, output_dir=RPB_output_dir, basin_shp=RPB_shp,
                                    ref_raster=WestUS_raster, resolution=model_res,
                                    skip_processing=skip_process_rpb_data)

        # # # # #  STEP 3 # # # # #
        # # Compile growing season netGW and annual pumping in a dataframe
        print('# # # # #  STEP 3 # # # # #')

        basin_pumping_mm_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/pumping_mm'
        basin_pumping_AF_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/pumping_AF_raster'
        RPB_csv = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/CO_RPB_netGW_pumping.csv'

        compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_output_dir,
                                           basin_pumping_mm_dir=basin_pumping_mm_dir,
                                           basin_pumping_AF_dir=basin_pumping_AF_dir,
                                           output_csv=RPB_csv, skip_processing=skip_process_rpb_data)

        annual_csv = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/CO_RPB_annual.csv'
        aggregate_pixelCSV_to_annualCSV(pixel_csv=RPB_csv, output_annual_csv=annual_csv)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Harquahala INA, Arizona
    if not skip_process_hqr_data:
        print('Processing netGW, pumping dataset and netGW-pumping dataframe for Harquahala_INA, AZ...')

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for Harquahala_INA
        print('# # # # #  STEP 1 # # # # #')

        years = [2016, 2017, 2018, 2019, 2020]
        Harquahala_INA_shp = '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp'
        netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/netGW_Harquahala_INA_AZ'
        irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        basin_irrig_fraction_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/irrig_frac'

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=Harquahala_INA_shp,
                                      netGW_input_dir=netGW_irrigation_dir,
                                      basin_netGW_output_dir=basin_netGW_output_dir,
                                      irr_frac_input_dir=irrig_fraction_dir,
                                      basin_irr_frac_output_dir=basin_irrig_fraction_output_dir,
                                      resolution=model_res, skip_processing=skip_process_hqr_data)

        # # # # #  STEP 2 # # # # #
        # # Converting annual pumping shapefile (unit AF) to mm raster
        print('# # # # #  STEP 2 # # # # #')

        pumping_shp = '../../Data_main/Pumping/Arizona/AZ_GW_Harquahala.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        Harquahala_INA_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ'

        pumping_AF_pts_to_mm_raster(years=years, irrigated_fraction_dir=irrig_fraction_dir,
                                    pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                                    year_attr=year_attr, output_dir=Harquahala_INA_output_dir, basin_shp=Harquahala_INA_shp,
                                    ref_raster=WestUS_raster, resolution=model_res,
                                    skip_processing=skip_process_hqr_data)

        # # # # #  STEP 3 # # # # #
        # # Compile growing season netGW and annual pumping in a dataframe
        print('# # # # #  STEP 3 # # # # #')

        basin_pumping_mm_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/pumping_mm'
        basin_pumping_AF_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/pumping_AF_raster'
        Harquahala_INA_csv = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/AZ_Harquahala_INA_netGW_pumping.csv'

        compiled_csv = compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_output_dir,
                                                          basin_pumping_mm_dir=basin_pumping_mm_dir,
                                                          output_csv=Harquahala_INA_csv,
                                                          basin_pumping_AF_dir=basin_pumping_AF_dir,
                                                          skip_processing=skip_process_hqr_data)

        annual_csv = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/AZ_Harquahala_INA_annual.csv'
        aggregate_pixelCSV_to_annualCSV(pixel_csv=Harquahala_INA_csv, output_annual_csv=annual_csv)

        # # # # #  STEP 4 # # # # #
        # # Extracting predicted pumping from Majumdar et al. 2022
        print('# # # # #  STEP 4 # # # # #')
        AZ_pumping_Majumadar_2022_dir = '../../Data_main/Raster_data/AZ_predictions_Majumdar et al. 2022/Postprocessed'
        AZ_pumping_Majumadar_2022_resampled_dir = '../../Data_main/Raster_data/AZ_predictions_Majumdar et al. 2022/Postprocessed_resampled'
        Harquahala_INA_csv_updated = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/AZ_Harquahala_INA_netGW_pumping_updated.csv'

        extract_pumping_estimate_with_lat_lon(years, input_csv=compiled_csv,
                                              input_data_dir=AZ_pumping_Majumadar_2022_dir,
                                              resampled_output_dir=AZ_pumping_Majumadar_2022_resampled_dir,
                                              output_csv=Harquahala_INA_csv_updated,
                                              ref_rater=WestUS_raster, resolution=model_res,
                                              skip_processing=skip_process_hqr_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Douglas AMA, Arizona
    if not skip_process_doug_data:
        print('Processing netGW, pumping dataset and netGW-pumping dataframe for Douglas_AMA, AZ...')

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for Douglas_AMA
        print('# # # # #  STEP 1 # # # # #')

        years = [2016, 2017, 2018, 2019, 2020]
        Douglas_AMA_shp = '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp'
        netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/netGW_Douglas_AMA_AZ'
        irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        basin_irrig_fraction_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/irrig_frac'

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=Douglas_AMA_shp,
                                      netGW_input_dir=netGW_irrigation_dir,
                                      basin_netGW_output_dir=basin_netGW_output_dir,
                                      irr_frac_input_dir=irrig_fraction_dir,
                                      basin_irr_frac_output_dir=basin_irrig_fraction_output_dir,
                                      resolution=model_res, skip_processing=skip_process_doug_data)

        # # # # #  STEP 2 # # # # #
        # # Converting annual pumping shapefile (unit AF) to mm raster
        print('# # # # #  STEP 2 # # # # #')

        pumping_shp = '../../Data_main/Pumping/Arizona/AZ_GW_Douglas.shp'
        pumping_attr_AF = 'AF_pumped'
        year_attr = 'Year'
        Douglas_AMA_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ'

        pumping_AF_pts_to_mm_raster(years=years, irrigated_fraction_dir=irrig_fraction_dir,
                                    pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                                    year_attr=year_attr, output_dir=Douglas_AMA_output_dir, basin_shp=Douglas_AMA_shp,
                                    ref_raster=WestUS_raster, resolution=model_res,
                                    skip_processing=skip_process_doug_data)

        # # # # #  STEP 3 # # # # #
        # # Compile growing season netGW and annual pumping in a dataframe
        print('# # # # #  STEP 3 # # # # #')

        basin_pumping_mm_dir = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/pumping_mm'
        basin_pumping_AF_dir = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/pumping_AF_raster'
        Douglas_AMA_csv = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/AZ_Douglas_AMA_netGW_pumping.csv'

        compiled_csv = compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_output_dir,
                                                          basin_pumping_mm_dir=basin_pumping_mm_dir,
                                                          basin_pumping_AF_dir=basin_pumping_AF_dir,
                                                          output_csv=Douglas_AMA_csv,
                                                          skip_processing=skip_process_doug_data)

        annual_csv = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/AZ_Douglas_AMA_annual.csv'
        aggregate_pixelCSV_to_annualCSV(pixel_csv=Douglas_AMA_csv, output_annual_csv=annual_csv)

        # # # # #  STEP 4 # # # # #
        # # Extracting predicted pumping from Majumdar et al. 2022
        print('# # # # #  STEP 4 # # # # #')

        AZ_pumping_Majumadar_2022_dir = '../../Data_main/Raster_data/AZ_predictions_Majumdar et al. 2022/Postprocessed'
        AZ_pumping_Majumadar_2022_resampled_dir = '../../Data_main/Raster_data/AZ_predictions_Majumdar et al. 2022/Postprocessed_resampled'
        Douglas_AMA_csv_updated = '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/AZ_Douglas_AMA_netGW_pumping_updated.csv'

        extract_pumping_estimate_with_lat_lon(years, input_csv=compiled_csv,
                                              input_data_dir=AZ_pumping_Majumadar_2022_dir,
                                              resampled_output_dir=AZ_pumping_Majumadar_2022_resampled_dir,
                                              output_csv=Douglas_AMA_csv_updated,
                                              ref_rater=WestUS_raster, resolution=model_res,
                                              skip_processing=skip_process_doug_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For Diamond Valley, Nevada
    if not skip_process_dv_data:
        print('Processing netGW, pumping dataset and netGW-pumping dataframe for Diamond Valley, NV...')

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for Diamond Valley
        print('# # # # #  STEP 1 # # # # #')

        years = [2016, 2017, 2018, 2019, 2020]
        Diamond_Valley_shp = '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp'
        netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
        basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Diamond_Valley_NV/netGW_Diamond_Valley_NV'
        irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        basin_irrig_fraction_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/irrig_frac'

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=Diamond_Valley_shp,
                                      netGW_input_dir=netGW_irrigation_dir,
                                      basin_netGW_output_dir=basin_netGW_output_dir,
                                      irr_frac_input_dir=irrig_fraction_dir,
                                      basin_irr_frac_output_dir=basin_irrig_fraction_output_dir,
                                      resolution=model_res, skip_processing=skip_process_dv_data)

        # # # # #  STEP 2 # # # # #
        # # Aggregating netGW and pumping data annually
        print('# # # # #  STEP 2 # # # # #')

        pumping_data = '../../Data_main/Pumping/Nevada/joined_data/dv_joined_et_pumping_data_all.csv'
        Diamond_Valley_csv = '../../Data_main/Raster_data/NetGW_irrigation/Diamond_Valley_NV/NV_Diamond_Valley_annual.csv'

        aggregate_netGW_pumping_to_annual(years=years, basin_netGW_dir=basin_netGW_output_dir,
                                          pumping_csv=pumping_data, pump_attr='pumping_AF',
                                          output_csv=Diamond_Valley_csv,
                                          skip_processing=skip_process_dv_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # For annual AF pumping and netGW compilation
    if not skip_compile_AF_annual_data:
        print('Compiling annual pumping and net GW estimates for all basin in dataframe...')

        basin_annual_csvs = ['../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/KS_GMD4_annual.csv',
                             '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/KS_GMD3_annual.csv',
                             '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/CO_RPB_annual.csv',
                             '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/AZ_Harquahala_INA_annual.csv',
                             '../../Data_main/Raster_data/NetGW_irrigation/Douglas_AMA_AZ/AZ_Douglas_AMA_annual.csv',
                             '../../Data_main/Raster_data/NetGW_irrigation/Diamond_Valley_NV/NV_Diamond_Valley_annual.csv']
        output_csv = '../../Data_main/Raster_data/NetGW_irrigation/annual_AF_pumping_netGW_BOI.csv'

        compile_annual_AF_pumping_netGW_all_basins(annual_csv_list=basin_annual_csvs, output_csv=output_csv)
