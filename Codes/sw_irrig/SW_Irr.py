import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.sw_irrig.SW_utils import create_canal_coverage_raster, estimate_sw_mm_HUC12, \
    distribute_SW_irrigation_to_pixels

WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

# # # # #  STEP 1 # # # # #
# # create yearly canal coverage raster compiling irrigated cropET (growing season) and canal coverage 2km buffer
print('# # # # #  STEP 1 # # # # #')

years_list = [2016, 2017, 2018, 2019, 2020]
canal_cover_hapefile = '../../Data_main/shapefiles/Surface_water_shapes/canals_buffered_1km_epsg_4269.shp'
original_irrig_cropET_grow_season_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
canal_cover_output_dir = '../../Data_main/Raster_data/Canal_coverage/canal_coverage'
irrigated_cropET_canal_cover_output_dir = '../../Data_main/Raster_data/Canal_coverage/irrigated_cropET_with_canal_cover'

skip_canal_coverage_creation = False  ######

create_canal_coverage_raster(years_list=years_list, canal_shapefile=canal_cover_hapefile,
                             irrig_cropET_grow_season_dir=original_irrig_cropET_grow_season_dir,
                             canal_coverage_output_dir=canal_cover_output_dir,
                             irrigated_CropET_with_canal_coverage_dir=irrigated_cropET_canal_cover_output_dir,
                             skip_processing=skip_canal_coverage_creation)


# # # # #  STEP 2 # # # # #
# # estimate sum irrigated cropET (that have canal cover) and number of irrigated pixels with canal coverage for each HUC12
print('# # # # #  STEP 2 # # # # #')

HUC12_shapefile_with_tot_SW_irrigation = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_SW.shp'
irrig_cropET_canal_cover_dir = '../../Data_main/Raster_data/Canal_coverage/irrigated_cropET_with_canal_cover'
HUC12_output_shapefile = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_Annual_SW_irrig_cropET.shp'

skip_estimate_sw_mm_data = False  ######

estimate_sw_mm_HUC12(years_list=years_list, HUC12_input_shapefile=HUC12_shapefile_with_tot_SW_irrigation,
                     irrigated_CropET_with_canal_coverage_dir=irrig_cropET_canal_cover_dir,
                     HUC12_output_shapefile=HUC12_output_shapefile,
                     skip_precessing=skip_estimate_sw_mm_data)


# # # # #  STEP 3 # # # # #
# # distribute HUC12 level sw irrigation (mm, area averaged) to irrigated pixels
print('# # # # #  STEP 3 # # # # #')

HUC12_shapefile_with_tot_SW_irrigation = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_Annual_SW_irrig_cropET.shp'
original_irrig_cropET_grow_season_dir = '../../Data_main/Raster_data/Canal_coverage/irrigated_cropET_with_canal_cover'
sw_dist_outdir = '../../Data_main/Raster_data/SW_irrigation'
skip_sw_dist = False  ######

distribute_SW_irrigation_to_pixels(years_list=years_list, HUC12_shapefile=HUC12_shapefile_with_tot_SW_irrigation,
                                   irrigated_CropET_with_canal_coverage_dir=original_irrig_cropET_grow_season_dir,
                                   sw_dist_outdir=sw_dist_outdir,
                                   ref_raster=WestUS_raster, skip_processing=skip_sw_dist)