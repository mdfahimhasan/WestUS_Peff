import os
import pandas as pd
from glob import glob
from osgeo import gdal

from Codes.utils.system_ops import makedirs
from Codes.utils.vector_ops import clip_vector
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster, \
    clip_resample_reproject_raster, make_lat_lon_array_from_raster


no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def clip_netGW_SW_for_basin(years, basin_shp, netGW_input_dir, SW_input_dir,
                            basin_netGW_output_dir, basin_SW_output_dir,
                            resolution=model_res):
    """
    Clip netGW and SW consumptive use datasets for a basin.

    :param years: List of years_list to process data.
    :param basin_shp: Filepath of basin shapefile.
    :param netGW_input_dir: Directory path of Western US netGW estimates.
    :param SW_input_dir: Directory path of Western US SW consumptive use estimates.
    :param basin_netGW_output_dir: Output directory path to save the clipped netGW estimates for the basin.
    :param basin_SW_output_dir: Output directory path to save the clipped SW consumptive use values for the basin.
    :param resolution: model resolution.

    :return: None.
    """

    for year in years:
        print(f'Clip netGW and SW for {year}...')

        # netGW
        netGW_raster = glob(os.path.join(netGW_input_dir, f'*{year}*.tif'))[0]

        clip_resample_reproject_raster(input_raster=netGW_raster, input_shape=basin_shp,
                                       output_raster_dir=basin_netGW_output_dir,
                                       keyword=' ', raster_name=f'netGW_Irr_{year}.tif',
                                       clip=True, resample=False, clip_and_resample=False,
                                       targetaligned=True, resample_algorithm='near',
                                       resolution=resolution,
                                       crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)

        # SW consumptive use
        SW_raster = glob(os.path.join(SW_input_dir, f'*{year}*.tif'))[0]

        clip_resample_reproject_raster(input_raster=SW_raster, input_shape=basin_shp,
                                       output_raster_dir=basin_SW_output_dir,
                                       keyword=' ', raster_name=f'SW_Irr_{year}.tif',
                                       clip=True, resample=False, clip_and_resample=False,
                                       targetaligned=True, resample_algorithm='near',
                                       resolution=resolution,
                                       crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)

# def clip_USGS_data():


def compile_basin_netGW_SW(years, basin_netGW_dir, basin_SW_dir, basin_name, output_csv):
    """
    Compiling pixel-wise annual netGW and pumping data for a basin.

    :param years: List of years_list to process data.
    :param basin_netGW_dir: Basin netGW directory.
    :param basin_pumping_mm_dir: Basin pumping (in mm) directory.
    :param basin_water_yr_precip_dir: Basin pumping (in AF) directory.
    :param output_csv: Filepath of output csv.

    :return:  Filepath of output csv.
    """
    makedirs([os.path.dirname(output_csv)])

    print(f'Compiling netGW vs SW dataframe...')

    # empty dictionary with to store data
    extract_dict = {'year': [], 'netGW_mm': [], 'SW_mm': []}

    # lopping through each year and storing data in a list
    for year in years:
        netGW_data = glob(os.path.join(basin_netGW_dir, f'*{year}*.tif'))[0]
        SW_data = glob(os.path.join(basin_SW_dir, f'*{year}*.tif'))[0]

        netGW_arr = read_raster_arr_object(netGW_data, get_file=False).flatten()
        SW_arr = read_raster_arr_object(SW_data, get_file=False).flatten()

        year_list = [year] * len(netGW_arr)

        extract_dict['year'].extend(year_list)
        extract_dict['netGW_mm'].extend(list(netGW_arr))
        extract_dict['SW_mm'].extend(list(SW_arr))

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)
    df.dropna(inplace=True)

    # groupby operation to calculate stats
    df_agg = df.groupby('year')[['SW_mm', 'netGW_mm']].agg(['sum']).reset_index()

    # # replacing multi-index column names with single column name
    col_replace_dict = {'sum': '_sum'}
    new_cols = ['year']

    for col in df_agg.columns[1:]:
        first_index = col[0]
        second_index = col[-1]
        new_col_name = first_index + col_replace_dict[second_index]
        new_cols.append(new_col_name)

    df_agg.columns = new_cols
    df_agg['SW%'] = df_agg['SW_mm_sum'] * 100 / (df_agg['SW_mm_sum'] + df_agg['netGW_mm_sum'])
    df_agg['GW%'] = df_agg['netGW_mm_sum'] * 100 / (df_agg['SW_mm_sum'] + df_agg['netGW_mm_sum'])

    df_agg.to_csv(output_csv, index=False)


# For Central Valley

basin_code = 'cv'
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Central_Valley.shp'
westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
westUS_SW_dir = '../../Data_main/Raster_data/SW_irrigation'
basin_netGW_output_dir = f'../../Data_main/results_eval/irrig_perc/{basin_code}/netGW'
basin_SW_output_dir = f'../../Data_main/results_eval/irrig_perc/{basin_code}/SW'

clip_netGW_SW_for_basin(years=years, basin_shp=basin_shp,
                        netGW_input_dir=westUS_netGW_dir, SW_input_dir=westUS_SW_dir,
                        basin_netGW_output_dir=basin_netGW_output_dir, basin_SW_output_dir=basin_SW_output_dir,
                        resolution=model_res)

output_csv = f'../../Data_main/results_eval/irrig_perc/{basin_code}/irrig_split.csv'
compile_basin_netGW_SW(years, basin_netGW_dir=basin_netGW_output_dir,
                       basin_SW_dir=basin_SW_output_dir, basin_name=basin_code,
                       output_csv=output_csv)

# # For Snake River Basin

basin_code = 'srb'
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
basin_shp = '../../Data_main/shapefiles/Basins_of_interest/Snake_River_Basin.shp'
westUS_netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
westUS_SW_dir = '../../Data_main/Raster_data/SW_irrigation'
basin_netGW_output_dir = f'../../Data_main/results_eval/irrig_perc/{basin_code}/netGW'
basin_SW_output_dir = f'../../Data_main/results_eval/irrig_perc/{basin_code}/SW'

clip_netGW_SW_for_basin(years=years, basin_shp=basin_shp,
                        netGW_input_dir=westUS_netGW_dir, SW_input_dir=westUS_SW_dir,
                        basin_netGW_output_dir=basin_netGW_output_dir, basin_SW_output_dir=basin_SW_output_dir,
                        resolution=model_res)

output_csv = f'../../Data_main/results_eval/irrig_perc/{basin_code}/irrig_split.csv'
compile_basin_netGW_SW(years, basin_netGW_dir=basin_netGW_output_dir,
                       basin_SW_dir=basin_SW_output_dir, basin_name=basin_code,
                       output_csv=output_csv)