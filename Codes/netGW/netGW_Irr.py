import sys

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.netGW.netGW_utils import estimate_netGW_Irr

# estimating netGW (coverage WestUS, need to refine by adding SW irrigation)
years = [2016, 2017, 2018, 2019, 2020]
model_version = 'v6'
effective_precip = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season'
irrigated_cropET = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
irrigated_fraction = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
sw_irrigation_dir = '../../Data_main/Raster_data/SW_irrigation'
netGW_irrigation_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'

skip_westUS_netGW_processing = True
estimate_netGW_Irr(years_list=years, effective_precip_dir=effective_precip, irrigated_cropET_dir=irrigated_cropET,
                    irrigated_fraction_dir=irrigated_fraction, sw_irrigation_dist_dir=sw_irrigation_dir,
                    output_dir=netGW_irrigation_output_dir, skip_processing=skip_westUS_netGW_processing)

