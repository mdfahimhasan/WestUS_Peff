import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.sw_irrig.SW_utils import estimate_sw_mm_HUC12, distribute_SW_consmp_use_to_pixels

WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'

if __name__ == '__main__':
    skip_estimate_sw_mm_data = False  ######
    skip_sw_dist = False  ######

    # # # # #  STEP 1 # # # # #
    # # estimate sum irrigated cropET (that have canal cover) and number of irrigated pixels with canal coverage for each HUC12
    print('# # # # #  STEP 1 # # # # #')

    years_list = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                  2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                  2016, 2017, 2018, 2019, 2020]
    HUC12_shapefile_with_tot_SW_irrigation = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Annual_SW.shp'
    irrig_cropET_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season_v2'
    HUC12_output_shapefile = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_Annual_SW_irrig_cropET.shp'

    estimate_sw_mm_HUC12(years_list=years_list, HUC12_input_shapefile=HUC12_shapefile_with_tot_SW_irrigation,
                         irrigated_CropET_with_canal_coverage_dir=irrig_cropET_dir,
                         HUC12_output_shapefile=HUC12_output_shapefile,
                         skip_precessing=skip_estimate_sw_mm_data)

    # # # # #  STEP 2 # # # # #
    # # distribute HUC12 level sw consumptive use (mm, area averaged) to irrigated pixels
    print('# # # # #  STEP 2 # # # # #')

    HUC12_shapefile_with_tot_SW_irrigation = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_Annual_SW_irrig_cropET.shp'
    HUC12_Irr_eff_shapefile = '../../Data_main/USGS_water_use_data/USGS_new_wateruse_data_HUC12/HUC12_WestUS_with_Irr_Eff.shp'
    original_irrig_cropET_grow_season_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season_v2'
    sw_dist_outdir = '../../Data_main/Raster_data/SW_irrigation'

    distribute_SW_consmp_use_to_pixels(years_list=years_list, HUC12_shapefile=HUC12_shapefile_with_tot_SW_irrigation,
                                       HUC12_Irr_eff_shapefile=HUC12_Irr_eff_shapefile,
                                       irrigated_CropET_with_canal_coverage_dir=original_irrig_cropET_grow_season_dir,
                                       sw_dist_outdir=sw_dist_outdir,
                                       ref_raster=WestUS_raster, skip_processing=skip_sw_dist)
