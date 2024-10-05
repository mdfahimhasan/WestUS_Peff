import os
import sys
import numpy as np
from glob import glob

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def estimate_netGW_Irr(years_list, effective_precip_dir_pp, irrigated_cropET_dir,
                       irrigated_fraction_dir, sw_cnsmp_use_dir, output_dir,
                       ref_raster=WestUS_raster, skip_processing=False):
    """
    Estimate growing season (annual) net groundwater irrigation (consumptive groundwater use by crops or ET_gw) for
    the Western US compiling growing season irrigated cropET, growing season effective precipitation, and growing
    season surface water irrigation that has been distributed).

    :param years_list: A list of years_list to process data for.
    :param effective_precip_dir_pp: Directory path for growing season effective precipitation (post processed version).
                                (source: effective precipitation ML model)
    :param irrigated_cropET_dir: Directory path for growing season irrigated cropET.
                                (source: OpenET ensemble overlaid with irrigation masks)
    :param irrigated_fraction_dir: Directory path for irrigated cropland fraction in 2 km pixels.
                                (source: OpenET ensemble overlaid with irrigation masks)
    :param sw_cnsmp_use_dir: Directory path for growing season distributed surface water consumptive use.
                                (source: USGS HUC12-level surface water irrigation dataset)
    :param output_dir: Output directory to save the growing season netGW datasets.
    :param ref_raster : Western US reference raster.
    :param skip_processing: Set to True if want to skip this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        for year in years_list:
            print(f'Estimating growing season netGW for {year}...')

            # loading effective precipitation, irrigated cropET, irrigated fraction,
            # and surface water irrigation datasets
            eff_precip = glob(os.path.join(effective_precip_dir_pp, f'*{year}*.tif'))[0]
            irrigated_cropET = glob(os.path.join(irrigated_cropET_dir, f'*{year}*.tif'))[0]
            irrigated_fraction = glob(os.path.join(irrigated_fraction_dir, f'*{year}*.tif'))[0]
            sw_cnsmp_data = glob(os.path.join(sw_cnsmp_use_dir, f'*{year}*.tif'))[0]

            eff_precip_arr = read_raster_arr_object(eff_precip, get_file=False)
            irrigated_cropET_arr = read_raster_arr_object(irrigated_cropET, get_file=False)
            irrigated_frac_arr = read_raster_arr_object(irrigated_fraction, get_file=False)
            sw_cnsmp_use_arr = read_raster_arr_object(sw_cnsmp_data, get_file=False)

            # # estimating growing season net ET (SW and GW) irrigation
            net_et_irrig = np.where(~np.isnan(eff_precip_arr), irrigated_cropET_arr - eff_precip_arr, -9999)

            # the processed net ET irrigation estimates are averaged over only irrigated areas in a pixel.
            # before subtracting sw irrigation from this value to get netGW_irrig, the net_et_irrig need to be area
            # averaged for 2km pixel. This will lead to area-averaged netGW estimate which can be compared to area-averaged pumping.
            # multiplying with irrigated fraction will give the 2km pixel averaged net et irrigation estimates
            net_et_irrig_aa = np.where(~np.isnan(net_et_irrig), net_et_irrig * irrigated_frac_arr, -9999)

            # netGW estimation
            # netGW over irrigated cropland = irrigated cropET - effective precipitation - surface water irrigation
            net_gw_irrig = np.where(~np.isnan(net_et_irrig_aa), net_et_irrig_aa - sw_cnsmp_use_arr, -9999)

            # in case net GW is < 0 (surface water irrigation or effective precipitation is higher than
            # irrigated cropET), netGW is assigned to 0
            net_gw_irrig = np.where(net_gw_irrig >= 0, net_gw_irrig, -9999)

            # assigning 0 to all non-irrigated pixels inside the landmass of the Western US
            # using reference raster
            net_gw_irrig = np.where((net_gw_irrig == -9999) & (ref_arr == 0), 0, net_gw_irrig)

            output_raster = os.path.join(output_dir, f'netGW_Irr_{year}.tif')
            write_array_to_raster(net_gw_irrig, ref_file, ref_file.transform, output_raster)


if __name__ == '__main__':
    # estimating netGW (coverage WestUS, need to refine by adding SW irrigation)
    years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    model_version = 'v15'
    effective_precip = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_grow_season_scaled'
    irrigated_cropET = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season_v2'
    irrigated_fraction = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
    sw_irrigation_dir = '../../Data_main/Raster_data/SW_irrigation'
    netGW_irrigation_output_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'

    skip_westUS_netGW_processing = False  ######
    estimate_netGW_Irr(years_list=years, effective_precip_dir_pp=effective_precip,
                       irrigated_cropET_dir=irrigated_cropET,
                       irrigated_fraction_dir=irrigated_fraction, sw_cnsmp_use_dir=sw_irrigation_dir,
                       output_dir=netGW_irrigation_output_dir, skip_processing=skip_westUS_netGW_processing)
