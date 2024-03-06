import sys

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.netGW.netGW_utils import clip_netGW_for_basin, pumping_AF_pts_to_mm_raster, \
    compile_basin_df_for_netGW_pumping, extract_pumping_estimate_with_lat_lon

model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # For Groundwater Management District (GMD4), Kansas
print('Processing netGW, pumping dataset and netGW-pumpign dataframe for GMD4, KS...')

# # # # #  STEP 1 # # # # #
# # Clip growing season netGW for GMD4, KS
print('# # # # #  STEP 1 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
GMD4_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/netGW_GMD4_KS'

skip_clip_GMD4_netGW = False #####

clip_netGW_for_basin(years=years, basin_shp=GMD4_shp,
                     netGW_input_dir=netGW_irrigation_dir,
                     basin_netGW_output_dir=basin_netGW_output_dir,
                     resolution=model_res, skip_processing=skip_clip_GMD4_netGW)

# # # # #  STEP 2 # # # # #
# # Converting annual pumping shapefile (unit AF) to mm raster
print('# # # # #  STEP 2 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
pumping_shp = '../../Data_main/Pumping/Kansas/KS_GW_GMD4.shp'
pumping_attr_AF = 'WU_AF'
year_attr = 'Year'
GMD4_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp'
GMD4_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS'

skip_pump_dataset_processing_GMD4 = False  #####

pumping_AF_pts_to_mm_raster(years=years, pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                            year_attr=year_attr, output_dir=GMD4_output_dir, basin_shp=GMD4_shp, ref_raster=WestUS_raster,
                            resolution=model_res, skip_processing=skip_pump_dataset_processing_GMD4)

# # # # #  STEP 3 # # # # #
# # Compile growing season netGW and annual pumping in a dataframe
print('# # # # #  STEP 3 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
basin_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/netGW_GMD4_KS'
basin_pumping_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/pumping_mm'
GMD4_csv = '../../Data_main/Raster_data/NetGW_irrigation/GMD4_KS/KS_GMD4_netGW_pumping.csv'

skip_compile_GMD4_df = False  #####

compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                   basin_pumping_dir=basin_pumping_dir,
                                   output_csv=GMD4_csv, skip_processing=skip_compile_GMD4_df)

# # # # #  STEP 1 # # # # #
# # Clip growing season netGW for GMD3, KS
print('Processing netGW, pumping dataset and netGW-pumpign dataframe for GMD3, KS...')
print('# # # # #  STEP 1 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
GMD3_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/netGW_GMD3_KS'

skip_clip_GMD3_netGW = False  #####

clip_netGW_for_basin(years=years, basin_shp=GMD3_shp,
                     netGW_input_dir=netGW_irrigation_dir,
                     basin_netGW_output_dir=basin_netGW_output_dir,
                     resolution=model_res, skip_processing=skip_clip_GMD3_netGW)

# # # # #  STEP 2 # # # # #
# # Converting annual pumping shapefile (unit AF) to mm raster
print('# # # # #  STEP 2 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
pumping_shp = '../../Data_main/Pumping/Kansas/KS_GW_GMD3.shp'
pumping_attr_AF = 'AF_pumped'
year_attr = 'Year'
GMD3_shp = '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp'
GMD3_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS'

skip_pump_dataset_processing_GMD3 = False  #####

pumping_AF_pts_to_mm_raster(years=years, pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                            year_attr=year_attr, output_dir=GMD3_output_dir, basin_shp=GMD3_shp, ref_raster=WestUS_raster,
                            resolution=model_res, skip_processing=skip_pump_dataset_processing_GMD3)

# # # # #  STEP 3 # # # # #
# # Compile growing season netGW and annual pumping in a dataframe
print('# # # # #  STEP 3 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
basin_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/netGW_GMD3_KS'
basin_pumping_dir = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/pumping_mm'
GMD3_csv = '../../Data_main/Raster_data/NetGW_irrigation/GMD3_KS/KS_GMD3_netGW_pumping.csv'

skip_compile_GMD3_df = False  #####

compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                   basin_pumping_dir=basin_pumping_dir,
                                   output_csv=GMD3_csv, skip_processing=skip_compile_GMD3_df)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # For Republican River Basin (RPB), Colorado
print('Processing netGW, pumping dataset and netGW-pumpign dataframe for RPB, CO...')

# # # # #  STEP 1 # # # # #
# # Clip growing season netGW for RPB, CO
print('# # # # #  STEP 1 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
RPB_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/netGW_RPB_CO'

skip_clip_RPB_netGW = False  #####

clip_netGW_for_basin(years=years, basin_shp=RPB_shp,
                     netGW_input_dir=netGW_irrigation_dir,
                     basin_netGW_output_dir=basin_netGW_output_dir,
                     resolution=model_res, skip_processing=skip_clip_RPB_netGW)

# # # # #  STEP 2 # # # # #
# # Converting annual pumping shapefile (unit AF) to mm raster
print('# # # # #  STEP 2 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
pumping_shp = '../../Data_main/Pumping/Colorado/CO_GW_RPB.shp'
pumping_attr_AF = 'ann_amt'
year_attr = 'irr_year'
RPB_shp = '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp'
RPB_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO'

skip_pump_dataset_processing_RPB = False  #####

pumping_AF_pts_to_mm_raster(years=years, pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                            year_attr=year_attr, output_dir=RPB_output_dir, basin_shp=RPB_shp, ref_raster=WestUS_raster,
                            resolution=model_res, skip_processing=skip_pump_dataset_processing_RPB)

# # # # #  STEP 3 # # # # #
# # Compile growing season netGW and annual pumping in a dataframe
print('# # # # #  STEP 3 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
basin_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/netGW_RPB_CO'
basin_pumping_dir = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/pumping_mm'
RPB_csv = '../../Data_main/Raster_data/NetGW_irrigation/RPB_CO/CO_RPB_netGW_pumping.csv'

skip_compile_RPB_df = False  #####

compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                   basin_pumping_dir=basin_pumping_dir,
                                   output_csv=RPB_csv, skip_processing=skip_compile_RPB_df)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # For Harquahala INA (Harquahala_INA), Arizona
print('Processing netGW, pumping dataset and netGW-pumpign dataframe for Harquahala_INA, AZ...')

# # # # #  STEP 1 # # # # #
# # Clip growing season netGW for Harquahala_INA
print('# # # # #  STEP 1 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
Harquahala_INA_shp = '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp'
netGW_irrigation_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
basin_netGW_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/netGW_Harquahala_INA_AZ'

skip_clip_Harquahala_INA_netGW = False  #####

clip_netGW_for_basin(years=years, basin_shp=Harquahala_INA_shp,
                     netGW_input_dir=netGW_irrigation_dir,
                     basin_netGW_output_dir=basin_netGW_output_dir,
                     resolution=model_res, skip_processing=skip_clip_Harquahala_INA_netGW)

# # # # #  STEP 2 # # # # #
# # Converting annual pumping shapefile (unit AF) to mm raster
print('# # # # #  STEP 2 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
pumping_shp = '../../Data_main/Pumping/Arizona/AZ_GW_Harquahala.shp'
pumping_attr_AF = 'AF_pumped'
year_attr = 'Year'
Harquahala_INA_shp = '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp'
Harquahala_INA_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ'

skip_pump_dataset_processing_Harquahala_INA = False  #####

pumping_AF_pts_to_mm_raster(years=years, pumping_pts_shp=pumping_shp, pumping_attr_AF=pumping_attr_AF,
                            year_attr=year_attr, output_dir=Harquahala_INA_output_dir, basin_shp=Harquahala_INA_shp,
                            ref_raster=WestUS_raster, resolution=model_res,
                            skip_processing=skip_pump_dataset_processing_Harquahala_INA)

# # # # #  STEP 3 # # # # #
# # Compile growing season netGW and annual pumping in a dataframe
print('# # # # #  STEP 3 # # # # #')

years = [2016, 2017, 2018, 2019, 2020]
basin_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/netGW_Harquahala_INA_AZ'
basin_pumping_dir = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/pumping_mm'
Harquahala_INA_csv = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/AZ_Harquahala_INA_netGW_pumping.csv'

skip_compile_Harquahala_INA_df = False  #####

compiled_csv = compile_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                                  basin_pumping_dir=basin_pumping_dir,
                                                  output_csv=Harquahala_INA_csv,
                                                  skip_processing=skip_compile_Harquahala_INA_df)

AZ_pumping_Majumadar_2022_dir = '../../Data_main/Raster_data/AZ_predictions_Majumdar et al. 2022/Postprocessed'
AZ_pumping_Majumadar_2022_resampled_dir = '../../Data_main/Raster_data/AZ_predictions_Majumdar et al. 2022/Postprocessed_resampled'
Harquahala_INA_csv_updated = '../../Data_main/Raster_data/NetGW_irrigation/Harquahala_INA_AZ/AZ_Harquahala_INA_netGW_pumping_updated.csv'

skip_extract_Majumdar_estimate = False  #####

extract_pumping_estimate_with_lat_lon(years, input_csv=compiled_csv,
                                      input_data_dir=AZ_pumping_Majumadar_2022_dir,
                                      resampled_output_dir=AZ_pumping_Majumadar_2022_resampled_dir,
                                      output_csv=Harquahala_INA_csv_updated,
                                      ref_rater=WestUS_raster, resolution=model_res,
                                      skip_processing=skip_extract_Majumdar_estimate)