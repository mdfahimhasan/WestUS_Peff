import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.sw_irrig.sw_utils import create_canal_coverage_raster, estimate_sw_mm_HUC12, \
    distribute_SW_irrigation_to_pixels

WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

# # create yearly canal coverage raster
years_list = [2016, 2017, 2018, 2019, 2020]
canal_shapefile = '../../Data_main/shapefiles/Surface_water_shapes/canals_buffered_1km_epsg_4269.shp'
irrig_cropET_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
canal_cover_output_dir = '../../Data_main/Raster_data/Canal_coverage/canal_coverage'
irrigated_cropET_canal_cover_output_dir = '../../Data_main/Raster_data/Canal_coverage/irrigated_cropET'
skip_canal_coverage_creation = True

create_canal_coverage_raster(years_list=years_list, canal_shapefile=canal_shapefile,
                             irrig_cropET_grow_season_dir=irrig_cropET_dir,
                             canal_coverage_output_dir=canal_cover_output_dir,
                             irrigated_CropET_for_canal_coverage_dir=irrigated_cropET_canal_cover_output_dir,
                             skip_processing=skip_canal_coverage_creation)


# # estimate sum irrigated cropET and number of irrigated pixels with canal coverage for each HUC12
HUC12_shapefile = '../../Data_main/USGS_water_use_data/USGS_new_wateuse_data_HUC12/HUC12_WestUS_with_Annual_SW.shp'
irrig_cropET_dir = '../../Data_main/Raster_data/Canal_coverage/irrigated_cropET'
canal_coverage_dir = '../../Data_main/Raster_data/Canal_coverage/canal_coverage'
HUC12_output_shapefile = '../../Data_main/USGS_water_use_data/USGS_new_wateuse_data_HUC12/HUC12_WestUS_Annual_SW_irrig_cropET.shp'

skip_estimate_sw_mm_data = True

estimate_sw_mm_HUC12(years_list=years_list, HUC12_input_shapefile=HUC12_shapefile,
                     irrig_cropET_grow_season_dir=irrig_cropET_dir,
                     canal_coverage_dir=canal_coverage_dir,
                     HUC12_output_shapefile=HUC12_output_shapefile,
                     skip_precessing=skip_estimate_sw_mm_data)


# # distribute HUC12 level sw irrigation (mm, area averaged) to irrigated pixels
HUC12_shapefile = '../../Data_main/USGS_water_use_data/USGS_new_wateuse_data_HUC12/HUC12_WestUS_Annual_SW_irrig_cropET.shp'
irrig_cropET_dir = '../../Data_main/Raster_data/Canal_coverage/irrigated_cropET'
sw_dist_outdir = '../../Data_main/Raster_data/SW_irrigation'
skip_sw_dist = False

distribute_SW_irrigation_to_pixels(years_list=years_list, HUC12_shapefile=HUC12_shapefile,
                                   irrig_cropET_grow_season_dir=irrig_cropET_dir,
                                   sw_dist_outdir=sw_dist_outdir,
                                   ref_raster=WestUS_raster, skip_processing=skip_sw_dist)