import numpy as np
import pandas as pd
import geopandas as gpd
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster

WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'
no_data_value = -9999


def combine_irrigated_developed_raster(irrigated_raster='../../Data_main/Compiled_data/Irrigated_agri_2015.tif',
                                       developed_raster='../../Data_main/Compiled_data/USDA_developed_2015.tif',
                                       ref_raster=WestUS_raster,
                                       output_raster='../../scratch/net_gw_estimate/irrigated_developed_LU.tif'):
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    irrig_arr = read_raster_arr_object(irrigated_raster, get_file=False)
    dev_arr = read_raster_arr_object(developed_raster, get_file=False)

    combined_arr = np.where((irrig_arr > 0) | (dev_arr > 0), 1, ref_arr)

    write_array_to_raster(raster_arr=combined_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)

    return output_raster


def distribute_SW_use_per_pixel(combined_irrig_devp_raster='../../scratch/net_gw_estimate/irrigated_developed_LU.tif',
                                canal_coverage_raster='../../scratch/net_gw_estimate/canal_coverage.tif',
                                county_wateruse_csv='../../Data_main/USGS_water_use_data/WestUS_county_WaterUse.csv',
                                countyID_raster='../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif',
                                county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county.shp',
                                output_shapefile='../../scratch/net_gw_estimate/sw_distribution.shp'):

    landUse_arr = read_raster_arr_object(combined_irrig_devp_raster, get_file=False)
    canal_arr = read_raster_arr_object(canal_coverage_raster, get_file=False)

    # counting number of pixels that are irrigated/developed and inside canal buffer (i.e., number of pixels possibly
    # getting SW irrigation/supply in each county
    county_arr = read_raster_arr_object(countyID_raster, get_file=False)
    unique_fips, counts = np.unique(county_arr[~np.isnan(county_arr) & (landUse_arr == 1) & (canal_arr == 1)],
                                    return_counts=True)
    count_dict = {'fips': unique_fips, 'pixelCount': counts}

    # Creating a dataframe which have total counts of irrigated cropland and developed pixels in each county
    count_df = pd.DataFrame(count_dict)

    county_gdf = gpd.read_file(county_shape)
    county_gdf = county_gdf.drop(columns=['COUNTYNS', 'AFFGEOID', 'GEOID', 'NAME', 'LSAD', 'ALAND', 'AWATER'])

    wateruse_df = pd.read_csv(county_wateruse_csv)

    county_gdf = county_gdf.merge(count_df, on=['fips'], how='left')  # 1st merge to add pixel count information
    county_gdf = county_gdf.merge(wateruse_df, on='fips', how='left')  # 2nd merge to add wateruse data

    # Distributing surface water use equally between irrigated/developed pixels that fall inside canal buffer
    # i.e., pixels possibly getting SW irrigation/supply
    county_gdf['SW_dist'] = county_gdf['sw_withdrawal'] / county_gdf['pixelCount']
    county_gdf['SW_dist'] = county_gdf['SW_dist'].fillna(0)

    county_gdf.to_file(output_shapefile)


def assign_SW_use(combined_irrig_devp_raster='../../scratch/net_gw_estimate/irrigated_developed_LU.tif',
                  canal_coverage_raster='../../scratch/net_gw_estimate/canal_coverage.tif',
                  countyID_raster='../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif',
                  sw_dist_shape='../../scratch/net_gw_estimate/sw_distribution.shp',
                  ref_raster=WestUS_raster, output_raster='../../scratch/net_gw_estimate/sw_distribution.tif'):

    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    combined_irrig_devp_arr, file = read_raster_arr_object(combined_irrig_devp_raster)
    canal_cover_arr = read_raster_arr_object(canal_coverage_raster, get_file=False)

    sw_dist_df = gpd.read_file(sw_dist_shape)
    countyID_arr = read_raster_arr_object(countyID_raster, get_file=False)

    # Assigning distributed surface water values to all cells of a county
    sw_dist_arr = None
    for i in sw_dist_df['fips']:
        sw_value = sw_dist_df[sw_dist_df['fips'] == i]['SW_dist']
        if sw_dist_arr is None:
            sw_dist_arr = np.where(countyID_arr == i, sw_value, 0)
        else:
            sw_dist_arr = np.where(countyID_arr == i, sw_value, sw_dist_arr)

    # Keeping distributed surface water values for cells with irrigation/development that fall in canal buffers
    final_sw_dist_arr = np.where((combined_irrig_devp_arr == 1) & (canal_cover_arr == 1), sw_dist_arr, ref_arr)

    write_array_to_raster(raster_arr=final_sw_dist_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)


def estimate_netGW_use(et_data='../../Data_main/Compiled_data/SSEBOP_ET_2015.tif',
                       sw_use_distribution='../../scratch/net_gw_estimate/sw_distribution.tif',
                       combined_irrig_devp_raster='../../scratch/net_gw_estimate/irrigated_developed_LU.tif',
                       ref_raster=WestUS_raster, output_raster='../../scratch/net_gw_estimate/net_GWuse_estimate.tif'):

    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    et_arr = read_raster_arr_object(et_data, get_file=False)
    sw_use_arr = read_raster_arr_object(sw_use_distribution, get_file=False)
    combined_irrig_devp_arr = read_raster_arr_object(combined_irrig_devp_raster, get_file=False)

    net_gw_use = et_arr - sw_use_arr
    net_gw_use = np.where(net_gw_use > 0, net_gw_use, 0)

    net_gw_arr = np.where(combined_irrig_devp_arr == 1, net_gw_use, ref_arr)

    write_array_to_raster(raster_arr=net_gw_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)


# converting canal buffer to raster
canal_buffer = '../../Data_main/shapefiles/Surface_water_shapes/canals_buffered_1km_epsg_4269.shp'
output_dir = '../../scratch/net_gw_estimate'

shapefile_to_raster(input_shape=canal_buffer, output_dir=output_dir, raster_name='canal_coverage.tif',
                    burnvalue=1, use_attr=False, alltouched=False)

# combining irrigated and developed pixels
combine_irrigated_developed_raster()

# Distributing SW use to irrigated/developed pixels that fall within canal buffer (potential SW irrigated pixels)
distribute_SW_use_per_pixel()

# Assigning SW use to irrigated/developed pixels that fall within canal buffer (potential SW irrigated pixels)
assign_SW_use()

# Estimate Net groundwater use using ET and surface wateruse distribution data
estimate_netGW_use()





