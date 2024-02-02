import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from osgeo import gdal

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster, \
    clip_resample_reproject_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def estimate_netGW(years_list, effective_precip_dir, irrigated_cropET_dir, output_dir,
                   skip_processing=False):
    if not skip_processing:
        makedirs([output_dir])

        for year in years_list:

            eff_precip = glob(os.path.join(effective_precip_dir, f'*{year}*.tif'))[0]
            irrigated_cropET = glob(os.path.join(irrigated_cropET_dir, f'*{year}*.tif'))[0]

            eff_precip_arr, file = read_raster_arr_object(eff_precip)
            irrigated_cropET_arr = read_raster_arr_object(irrigated_cropET, get_file=False)

            net_gw_irrig = np.where(~np.isnan(eff_precip_arr), irrigated_cropET_arr - eff_precip_arr, -9999)  # only trust this at GMD KS at this point
            net_gw_irrig[net_gw_irrig < 0] = -9999

            output_raster = os.path.join(output_dir, f'netGW_{year}.tif')
            write_array_to_raster(net_gw_irrig, file, file.transform, output_raster)


# estimating netGW (coverage WestUS, need to refine by adding SW irrigation)
years = [2016, 2017, 2018, 2019, 2020]
effective_precip ='../../Data_main/Raster_data/Effective_precip_prediction_WestUS/v5_grow_season'
irrigated_cropET = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_grow_season'
estimate_netGW(years_list=years, effective_precip_dir=effective_precip, irrigated_cropET_dir=irrigated_cropET,
               output_dir='../../scratch/netGW_KS/netGW_mm_initial', skip_processing=False)

# refine netGW estimate of KS using irrigated fraction
skip_apply_irrig_frac_on_netGW = False
if not skip_apply_irrig_frac_on_netGW:
    irrig_fraction_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
    netGW_dir = '../../scratch/netGW_KS/netGW_mm_initial'

    for year in years:
        irrig_fraction_data = glob(os.path.join(irrig_fraction_dir, f'*{year}*.tif'))[0]
        irrig_frac_arr = read_raster_arr_object(irrig_fraction_data, get_file=False)

        netGW_data = glob(os.path.join(netGW_dir, f'*{year}*.tif'))[0]
        netGW_arr, netGW_file = read_raster_arr_object(netGW_data)

        # multiplying with fraction will give the 2km pixel averaged
        # pixels with no pumping or zero pumping is assigned to 0 (need the no pumping info for GW models..)
        netGW_for2km = np.where(~np.isnan(netGW_arr), netGW_arr * irrig_frac_arr, 0)

        refined_netGW_raster = os.path.join('../../scratch/netGW_KS/netGW_mm_refined', f'netGW_{year}.tif')
        write_array_to_raster(netGW_for2km, netGW_file, netGW_file.transform, refined_netGW_raster)

        # clipping data only to GMD4, KS extent
        GMD4_shp = 'F:/WestUS_NetGW/scratch/GMD4.shp'
        final_netGW_GMD4_dir = '../../scratch/netGW_KS/KS_netGW_mm_final'
        output_raster_name = f'GMD4_netGW_{year}_mm.tif'
        clip_resample_reproject_raster(input_raster=refined_netGW_raster, input_shape=GMD4_shp,
                                       output_raster_dir=final_netGW_GMD4_dir,
                                       keyword=' ', raster_name=output_raster_name,
                                       clip=True, resample=False, clip_and_resample=False,
                                       targetaligned=True, resample_algorithm='near',
                                       resolution=model_res,
                                       crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)


# converting KS GW use data to raster
skip_process_KS_pumping = False
if not skip_process_KS_pumping:
    GMD4_pumping_pts = 'F:/WestUS_NetGW/Data_main/Data_Drivers_of_ET/shapes/KS_GW_GMD4.shp'
    KS_gdf = gpd.read_file(GMD4_pumping_pts)

    for year in years:
        KS_gdf_filtered = KS_gdf[KS_gdf['Year'] == year]
        KS_GW_filtered_shp = os.path.join('../../scratch/netGW_KS/KS_pumping/annual_pumping_shp', f'KS_GW_GMD4_{year}.shp')
        KS_gdf_filtered.to_file(KS_GW_filtered_shp)

        output_AF_raster = f'GMD4_pumping_{year}_AF.tif'
        pumping_AF_raster = shapefile_to_raster(input_shape=KS_GW_filtered_shp, output_dir='../../scratch/netGW_KS/KS_pumping/AF_raster',
                            raster_name=output_AF_raster, use_attr=True, attribute='WU_AF', add=True,
                            ref_raster=WestUS_raster, resolution=model_res)

        # area of a pixel
        area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2

        # converting pumping unit from AF to mm
        # pixels with no pumping or zero pumping is assigned to 0 (need the no pumping info for GW models..)
        pumping_AF_arr, file = read_raster_arr_object(pumping_AF_raster)
        pumping_mm_arr = np.where(~np.isnan(pumping_AF_arr), pumping_AF_arr * 1233481837548 / area_mm2_single_pixel, 0)

        output_mm_raster = os.path.join('../../scratch/netGW_KS/KS_pumping/mm_raster', f'GMD4_pumping_{year}_mm.tif')
        write_array_to_raster(pumping_mm_arr, file, file.transform, output_mm_raster)

        # clipping data only to GMD4, KS extent
        GMD4_shp = 'F:/WestUS_NetGW/scratch/GMD4.shp'
        final_KS_pumping_dir = '../../scratch/netGW_KS/KS_pumping/KS_pumping_mm_final'
        output_raster_name = f'GMD4_pumping_{year}_mm.tif'
        clip_resample_reproject_raster(input_raster=output_mm_raster, input_shape=GMD4_shp,
                                       output_raster_dir=final_KS_pumping_dir,
                                       keyword=' ', raster_name=output_raster_name,
                                       clip=True, resample=False, clip_and_resample=False,
                                       targetaligned=True, resample_algorithm='near',
                                       resolution=model_res,
                                       crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)

# make dataframe with KS netGW estimate and GW pumping (both area averaged for 2km)
skip_dataframe_compile = False

KS_netGW_dir = '../../scratch/netGW_KS/KS_netGW_mm_final'
KS_GW_pumping_dir = '../../scratch/netGW_KS/KS_pumping/KS_pumping_mm_final'

if not skip_dataframe_compile:
    extract_dict = {'year': [], 'KS_netGW_mm': [], 'KS_pump_mm': []}
    if not skip_dataframe_compile:
        for year in years:
            KS_netGW = glob(os.path.join(KS_netGW_dir, f'*{year}*.tif'))[0]
            KS_pump = glob(os.path.join(KS_GW_pumping_dir, f'*{year}*.tif'))[0]

            KS_netGW_arr = read_raster_arr_object(KS_netGW, get_file=False).flatten()
            KS_pump_arr = read_raster_arr_object(KS_pump, get_file=False).flatten()

            year_list = [year] * len(KS_pump_arr)

            extract_dict['year'].extend(year_list)
            extract_dict['KS_netGW_mm'].extend(list(KS_netGW_arr))
            extract_dict['KS_pump_mm'].extend(list(KS_pump_arr))

    df = pd.DataFrame(extract_dict)
    df = df.dropna().reset_index(drop=True)

    df.to_csv('../../scratch/netGW_KS/KS_GMD4_netGW_pumping.csv')

