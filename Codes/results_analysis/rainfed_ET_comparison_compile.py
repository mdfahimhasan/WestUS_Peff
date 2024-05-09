import sys

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from analysis_utils import compile_annual_irr_rainfed_ET

# # Reference resolution, shapes, and reference rasters
no_data_value = -9999

res = 0.01976293625031605786  # in deg, ~2 km

GMD_KS_shape = '../../Data_main/Data_Drivers_of_ET/ROIs/For_2km/KS_GMD.shp'
NE_ROI_shape = '../../Data_main/Data_Drivers_of_ET/ROIs/For_2km/NE_ROI.shp'
TX_ROI_shape = '../../Data_main/Data_Drivers_of_ET/ROIs/For_2km/TX_ROI.shp'

GMD_KS_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/gmd_KS_refraster.tif'
NE_ROI_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/NE_ROI_refraster.tif'
TX_ROI_refraster = '../../Data_main/Data_Drivers_of_ET/refrasters/2km/TX_ROI_refraster.tif'

# Original input data directories
USDA_CDL_yearly = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'
irrigated_cropland = '../../Data_main/Raster_data/Irrigated_cropland'
rainfed_cropland = '../../Data_main/Raster_data/Rainfed_cropland'
irrigated_cropET_grow_season = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly'
rainfed_cropET_grow_season = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly'

years_to_process = [2016, 2017, 2018, 2019, 2020]

if __name__== '__main__':
    # Process switches
    skip_process_KS_data = False
    skip_process_NE_data = False
    skip_process_TX_data = False

    # output csv paths
    KS_csv = '../../Data_main/results_eval/rainfed_cropET_compare/KS/KS_cropET.csv'
    NE_csv = '../../Data_main/results_eval/rainfed_cropET_compare/NE/NE_cropET.csv'
    TX_csv = '../../Data_main/results_eval/rainfed_cropET_compare/TX/TX_cropET.csv'

    # # For Groundwater Management District, Kansas
    compile_annual_irr_rainfed_ET(years=years_to_process, area_code='KS',
                                  area_shape=GMD_KS_shape, area_ref_raster=GMD_KS_refraster,
                                  cdl_input_dir=USDA_CDL_yearly,
                                  irrigated_cropland_input_dir=irrigated_cropland,
                                  rainfed_cropland_input_dir=rainfed_cropland,
                                  irrigated_cropET_input_dir=irrigated_cropET_grow_season,
                                  rainfed_cropET_input_dir=rainfed_cropET_grow_season,
                                  output_csv=KS_csv, skip_processing=skip_process_KS_data)

    # # For eastern HPA, Nebraska
    compile_annual_irr_rainfed_ET(years=years_to_process, area_code='NE',
                                  area_shape=NE_ROI_shape, area_ref_raster=NE_ROI_refraster,
                                  cdl_input_dir=USDA_CDL_yearly,
                                  irrigated_cropland_input_dir=irrigated_cropland,
                                  rainfed_cropland_input_dir=rainfed_cropland,
                                  irrigated_cropET_input_dir=irrigated_cropET_grow_season,
                                  rainfed_cropET_input_dir=rainfed_cropET_grow_season,
                                  output_csv=NE_csv, skip_processing=skip_process_NE_data)

    # # For southern HPA, Texas
    compile_annual_irr_rainfed_ET(years=years_to_process, area_code='TX',
                                  area_shape=TX_ROI_shape, area_ref_raster=TX_ROI_refraster,
                                  cdl_input_dir=USDA_CDL_yearly,
                                  irrigated_cropland_input_dir=irrigated_cropland,
                                  rainfed_cropland_input_dir=rainfed_cropland,
                                  irrigated_cropET_input_dir=irrigated_cropET_grow_season,
                                  rainfed_cropET_input_dir=rainfed_cropET_grow_season,
                                  output_csv=TX_csv, skip_processing=skip_process_TX_data)