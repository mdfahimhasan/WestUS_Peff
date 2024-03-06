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
    clip_resample_reproject_raster, make_lat_lon_array_from_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def estimate_netGW_Irr(years_list, effective_precip_dir, irrigated_cropET_dir,
                       irrigated_fraction_dir, sw_irrigation_dist_dir, output_dir,
                       ref_raster=WestUS_raster, skip_processing=False):
    """
    Estimate growing season (annual) net groundwater irrigation (consumptive groundwater use by crops or ET_gw) for
    the Western US compiling growing season irrigated cropET, growing season effective precipitation, and growing
    season surface water irrigation that has been distributed).

    :param years_list: A list of years to process data for.
    :param effective_precip_dir: Directory path for growing season effective precipitation.
                                (source: effective precipitation ML model)
    :param irrigated_cropET_dir: Directory path for growing season irrigated cropET.
                                (source: OpenET ensemble overlaid with irrigation masks)
    :param irrigated_fraction_dir: Directory path for irrigated cropland fraction in 2 km pixels.
                                (source: OpenET ensemble overlaid with irrigation masks)
    :param sw_irrigation_dist_dir: Directory path for growing season distributed surface water irrigation.
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
            eff_precip = glob(os.path.join(effective_precip_dir, f'*{year}*.tif'))[0]
            irrigated_cropET = glob(os.path.join(irrigated_cropET_dir, f'*{year}*.tif'))[0]
            irrigated_fraction = glob(os.path.join(irrigated_fraction_dir, f'*{year}*.tif'))[0]
            sw_irrig = glob(os.path.join(sw_irrigation_dist_dir, f'*{year}*.tif'))[0]

            eff_precip_arr = read_raster_arr_object(eff_precip, get_file=False)
            irrigated_cropET_arr = read_raster_arr_object(irrigated_cropET, get_file=False)
            irrigated_frac_arr = read_raster_arr_object(irrigated_fraction, get_file=False)
            sw_irrig_arr = read_raster_arr_object(sw_irrig, get_file=False)

            # # estimating growing season net ET (SW and GW) irrigation
            net_et_irrig = np.where(~np.isnan(eff_precip_arr), irrigated_cropET_arr - eff_precip_arr, -9999)

            # the processed net ET irrigation estimates are averaged over only irrigated areas in a pixel.
            # before subtracting sw irrigation from this value to get netGW_irrig, the net_et_irrig need to be area
            # averaged. This will lead to area-averaged netGW estimate which can be compared area-averaged pumping.
            # multiplying with irrigated fraction will give the 2km pixel averaged net et irrigation estimates
            net_et_irrig_aa = np.where(~np.isnan(net_et_irrig), net_et_irrig * irrigated_frac_arr, -9999)

            # netGW estimation
            # netGW over irrigated cropland = irrigated cropET - effective precipitation - surface water irrigation
            net_gw_irrig = np.where(~np.isnan(net_et_irrig_aa), net_et_irrig_aa - sw_irrig_arr, -9999)

            # in case net GW is < 0 (surface water irrigation or effective precipitation is higher than
            # irrigated cropET), netGW is assigned to 0
            net_gw_irrig = np.where(net_gw_irrig >= 0, net_gw_irrig, -9999)

            # assigning 0 to all non-irrigated pixels inside the landmass of the Western US
            # using reference raster
            net_gw_irrig = np.where((net_gw_irrig==-9999) & (ref_arr == 0), 0, net_gw_irrig)

            output_raster = os.path.join(output_dir, f'netGW_Irr_{year}.tif')
            write_array_to_raster(net_gw_irrig, ref_file, ref_file.transform, output_raster)


def clip_netGW_for_basin(years, basin_shp, netGW_input_dir, basin_netGW_output_dir,
                         resolution=model_res, skip_processing=False):
    if not skip_processing:
        for year in years:
            print(f'Clip growing season netGW for {year}...')

            netGW_raster = glob(os.path.join(netGW_input_dir, f'*{year}*.tif'))[0]

            clip_resample_reproject_raster(input_raster=netGW_raster, input_shape=basin_shp,
                                           output_raster_dir=basin_netGW_output_dir,
                                           keyword=' ', raster_name=f'netGW_Irr_{year}.tif',
                                           clip=True, resample=False, clip_and_resample=False,
                                           targetaligned=True, resample_algorithm='near',
                                           resolution=resolution,
                                           crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                           use_ref_width_height=False)


def pumping_AF_pts_to_mm_raster(years, pumping_pts_shp, pumping_attr_AF, year_attr, output_dir,
                                basin_shp, ref_raster=WestUS_raster, resolution=model_res,
                                skip_processing=False):
    if not skip_processing:
        # creating sub-directories
        annual_pump_shp_dir = os.path.join(output_dir, 'annual_pumping_shp')
        pumping_AF_dir = os.path.join(output_dir, 'pumping_AF_raster')
        pumping_mm_dir = os.path.join(output_dir, 'pumping_mm')

        makedirs([annual_pump_shp_dir, pumping_AF_dir, pumping_mm_dir])

        # loading pumping shapefile
        pumping_gdf = gpd.read_file(pumping_pts_shp)

        # looping by year and processing pumping shapefile to raster
        for year in years:
            print(f'Converting pumping AF shapefile to mm raster for {year}...')

            # filtering pumping dataset by year (unit Acre-ft) and saving it
            gdf_filtered = pumping_gdf[pumping_gdf[year_attr] == year]
            annual_filtered_shp = os.path.join(annual_pump_shp_dir, f'pumping_{year}.shp')
            gdf_filtered.to_file(annual_filtered_shp)

            # converting yearly pumping point dataset into yearly AF raster.
            # all pumping inside a 2 km pixel will be summed
            # the generated raster is for the whole Western US with 0 values outside the basin
            output_AF_raster = f'pumping_{year}_AF.tif'
            pumping_AF_raster = shapefile_to_raster(input_shape=annual_filtered_shp,
                                                    output_dir=os.path.join(pumping_AF_dir, 'interim'),
                                                    raster_name=output_AF_raster, use_attr=True,
                                                    attribute=pumping_attr_AF, add=True,
                                                    ref_raster=ref_raster, resolution=resolution)

            # clipping data only to required basin
            pumping_AF_raster = \
                clip_resample_reproject_raster(input_raster=pumping_AF_raster,
                                               input_shape=basin_shp, output_raster_dir=pumping_AF_dir,
                                               keyword=' ', raster_name=output_AF_raster,
                                               clip=True, resample=False, clip_and_resample=False,
                                               targetaligned=True, resample_algorithm='near',
                                               resolution=model_res,
                                               crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                               use_ref_width_height=False)

            # area of a 2 km pixel
            area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2

            # converting pumping unit from AF to mm
            # pixels with no pumping or zero pumping is assigned to 0 (need the no pumping info for GW models..)
            pumping_AF_arr, file = read_raster_arr_object(pumping_AF_raster)
            pumping_mm_arr = np.where(~np.isnan(pumping_AF_arr), pumping_AF_arr * 1233481837548 / area_mm2_single_pixel, 0)

            output_mm_raster = os.path.join(pumping_mm_dir, f'pumping_{year}_mm.tif')
            write_array_to_raster(pumping_mm_arr, file, file.transform, output_mm_raster)

    else:
        pass


def compile_basin_df_for_netGW_pumping(years, basin_netGW_dir, basin_pumping_dir,
                                       output_csv, skip_processing=False):
    if not skip_processing:
        makedirs([basin_pumping_dir])

        print(f'Compiling growing season netGW vs pumping dataframe...')

        # empty dictionary with to store data
        extract_dict = {'year': [], 'netGW_mm': [], 'pumping_mm': [], 'lat': [], 'lon': []}

        # lopping through each year and storing data in a list
        for year in years:
            netGW_data = glob(os.path.join(basin_netGW_dir, f'*{year}*.tif'))[0]
            pumping_data = glob(os.path.join(basin_pumping_dir, f'*{year}*.tif'))[0]

            netGW_arr = read_raster_arr_object(netGW_data, get_file=False).flatten()
            pump_arr = read_raster_arr_object(pumping_data, get_file=False).flatten()
            lon_arr, lat_arr = make_lat_lon_array_from_raster(netGW_data)
            lon_arr = lon_arr.flatten()
            lat_arr = lat_arr.flatten()

            year_list = [year] * len(pump_arr)

            extract_dict['year'].extend(year_list)
            extract_dict['netGW_mm'].extend(list(netGW_arr))
            extract_dict['pumping_mm'].extend(list(pump_arr))
            extract_dict['lon'].extend(list(lon_arr))
            extract_dict['lat'].extend(list(lat_arr))

        # converting dictionary to dataframe and saving to csv
        df = pd.DataFrame(extract_dict)
        df = df.dropna().reset_index(drop=True)

        df.to_csv(output_csv, index=False)

        return output_csv


def extract_pumping_estimate_with_lat_lon(years, input_csv, input_data_dir, resampled_output_dir,
                                          output_csv, ref_rater=WestUS_raster,
                                          resolution=model_res, skip_processing=False):
    if not skip_processing:
        df = pd.read_csv(input_csv)

        # empty dict to store extracted data
        result_dict = {'year': [], 'pumping_extracted_mm': [], 'lat': [], 'lon': []}

        makedirs([resampled_output_dir])
        # looping through each year and extracting values in each coordinate
        for year in years:
            pumping_data = glob(os.path.join(input_data_dir, f'*{year}*.tif'))[0]
            raster_name = os.path.basename(pumping_data)

            # resampling pumping data as it might be only for a region, but to extract data it need to be of
            # similar row, column as Western US extent
            resampled_pumping_data = clip_resample_reproject_raster(input_raster=pumping_data, input_shape=None,
                                           output_raster_dir=resampled_output_dir,
                                           keyword=' ', raster_name=raster_name,
                                           clip=False, resample=True, clip_and_resample=False,
                                           targetaligned=True, resample_algorithm='near',
                                           resolution=resolution,
                                           crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                           ref_raster=ref_rater, use_ref_width_height=True)

            # reading resampled pumping data
            pumping_arr, pumping_file = read_raster_arr_object(resampled_pumping_data)

            # filtering dataframe for a year
            df_filtered = df[df['year'] == year]

            # looping through each row to extract data with coordinate
            for idx, row in df_filtered.iterrows():
                x, y = pumping_file.index(row['lon'], row['lat'])
                value = pumping_arr[x, y]

                result_dict['year'].append(year)
                result_dict['lon'].append(row['lon'])
                result_dict['lat'].append(row['lat'])
                result_dict['pumping_extracted_mm'].append(value)

        # converting to dataframe
        result_df = pd.DataFrame(result_dict)

        # merging the dataframe
        df = df.merge(result_df, on=['year', 'lon', 'lat'])
        df = df.dropna()
        df.to_csv(output_csv, index=False)
