import os
import numpy as np
import pandas as pd
from glob import glob
from osgeo import gdal
import geopandas as gpd

from Codes.utils.system_ops import makedirs
from Codes.utils.vector_ops import clip_vector
from Codes.utils.ml_ops import create_train_test_monthly_dataframe
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster, \
    clip_resample_reproject_raster, make_lat_lon_array_from_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def clip_netGW_Irr_frac_for_basin(years, basin_shp, netGW_input_dir, basin_netGW_output_dir,
                                  resolution=model_res, irr_frac_input_dir=None,
                                  basin_irr_frac_output_dir=None):
    """
    Clip netGW and irrigated fraction datasets for a basin, Clipping irrigation fraction data is optional.

    :param years: List of years_list to process data.
    :param basin_shp: Filepath of basin shapefile.
    :param netGW_input_dir: Directory path of Western US netGW estimates.
    :param basin_netGW_output_dir: Output directory path to save the clipped netGW estimates for the basin.
    :param resolution: model resolution.
    :param irr_frac_input_dir: Directory path of Western US annual irrigation fraction datasets.
                               Default set to None to not do this step. To do this step, insert required directory path.
    :param basin_irr_frac_output_dir: Output directory path to save the clipped irrigation fraction estimates for the basin.

    :return: None.
    """

    for year in years:
        print(f'Clipping growing season netGW for {year}...')

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

        if irr_frac_input_dir is not None:
            print(f'Clipping irrigated fraction for {year}...')
            # irrigation fraction
            irr_frac_raster = glob(os.path.join(irr_frac_input_dir, f'*{year}*.tif'))[0]

            clip_resample_reproject_raster(input_raster=irr_frac_raster, input_shape=basin_shp,
                                           output_raster_dir=basin_irr_frac_output_dir,
                                           keyword=' ', raster_name=f'Irr_frac_{year}.tif',
                                           clip=True, resample=False, clip_and_resample=False,
                                           targetaligned=True, resample_algorithm='near',
                                           resolution=resolution,
                                           crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                           use_ref_width_height=False)


def pumping_AF_pts_to_raster(years, pumping_pts_shp, pumping_attr_AF,
                             year_attr, output_dir, basin_shp, ref_raster=WestUS_raster,
                             resolution=model_res):
    """
    Convert point scale (shapefile) groundwater pumping estimates to rasters in AF.
    For individual pixels (2km) sums up all the pumping values inside it.

    :param years: List of years_list to process data.
    :param pumping_pts_shp: Filepath of point shapefile with annual pumping estimates.
    :param pumping_attr_AF: Attribute in the point shapefile with pumping in AF values.
    :param year_attr: Attribute in the point shapefile with year.
    :param output_dir: Filepath of main output dir. Intermediate directories named 'pumping_AF_raster' and
                        'pumping_mm_raster' will be created automatically.
    :param basin_shp: Filepath of basin shapefile.
    :param ref_raster: Filepath of Western US reference raster.
    :param resolution: model resolution.

    :return: Raster directories' path with AF and mm pumping.
    """
    # creating sub-directories
    annual_pump_shp_dir = os.path.join(output_dir, 'annual_pumping_shp')
    pumping_AF_dir = os.path.join(output_dir, 'pumping_AF')
    interim_pumping_AF_dir = os.path.join(pumping_AF_dir, 'interim')
    pumping_mm_dir = os.path.join(output_dir, 'pumping_mm')
    interim_pumping_mm_dir = os.path.join(pumping_mm_dir, 'interim')

    makedirs([annual_pump_shp_dir, pumping_AF_dir, interim_pumping_AF_dir,
              pumping_mm_dir, interim_pumping_mm_dir])

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
                                                output_dir=interim_pumping_AF_dir,
                                                raster_name=output_AF_raster, use_attr=True,
                                                attribute=pumping_attr_AF, add=True,
                                                ref_raster=ref_raster, resolution=resolution)

        # converting pumping unit from AF to mm
        # pixels with no pumping or zero pumping is assigned to 0 (need the 'no pumping' info for GW models..)
        pumping_AF_arr, file = read_raster_arr_object(pumping_AF_raster)

        # area of a 2 km pixel
        area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2

        pumping_mm_arr = np.where(~np.isnan(pumping_AF_arr), pumping_AF_arr * 1233481837548 /
                                  area_mm2_single_pixel, 0)

        pumping_raster_name = f'pumping_{year}_mm.tif'
        pumping_mm_raster = os.path.join(interim_pumping_mm_dir, '', f'pumping_{year}_mm.tif')
        write_array_to_raster(pumping_mm_arr, file, file.transform, pumping_mm_raster)

        # # clipping data only to required basin
        # pumping AF data
        clip_resample_reproject_raster(input_raster=pumping_AF_raster,
                                       input_shape=basin_shp, output_raster_dir=pumping_AF_dir,
                                       keyword=' ', raster_name=output_AF_raster,
                                       clip=True, resample=False, clip_and_resample=False,
                                       targetaligned=True, resample_algorithm='near',
                                       resolution=model_res,
                                       crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)
        # pumping mm data
        clip_resample_reproject_raster(input_raster=pumping_mm_raster,
                                       input_shape=basin_shp, output_raster_dir=pumping_mm_dir,
                                       keyword=' ', raster_name=pumping_raster_name,
                                       clip=True, resample=False, clip_and_resample=False,
                                       targetaligned=True, resample_algorithm='near',
                                       resolution=model_res,
                                       crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)

    return pumping_AF_dir, pumping_mm_dir


def compile_pixelwise_basin_df_for_netGW_pumping(years, basin_netGW_dir, output_csv,
                                                 basin_pumping_mm_dir=None,
                                                 basin_pumping_AF_dir=None):
    """
    Compiling pixel-wise annual netGW and pumping data for a basin.

    :param years: List of years_list to process data.
    :param basin_netGW_dir: Basin netGW directory.
    :param output_csv: Filepath of output csv.
    :param basin_pumping_mm_dir: Basin pumping (in mm) directory.
                                 Default set to None to not incorporate pumping data (e.g., for Arizona)
    :param basin_pumping_AF_dir: Basin pumping (in AF) directory.
                                 Default set to None to not incorporate pumping data (e.g., for Arizona)

    :return:  Filepath of output csv.
    """
    if basin_pumping_mm_dir:
        makedirs([basin_pumping_mm_dir])

    print(f'Compiling growing season netGW vs pumping dataframe...')

    # empty dictionary with to store data
    extract_dict = {'year': [], 'netGW_mm': [], 'pumping_mm': [], 'pumping_AF': [],
                    'lat': [], 'lon': []}

    # lopping through each year and storing data in a list
    for year in years:
        netGW_data = glob(os.path.join(basin_netGW_dir, f'*{year}*.tif'))[0]
        netGW_arr = read_raster_arr_object(netGW_data, get_file=False).flatten()

        lon_arr, lat_arr = make_lat_lon_array_from_raster(netGW_data)
        lon_arr = lon_arr.flatten()
        lat_arr = lat_arr.flatten()

        year_list = [year] * len(netGW_arr)

        extract_dict['year'].extend(year_list)
        extract_dict['netGW_mm'].extend(list(netGW_arr))
        extract_dict['lon'].extend(list(lon_arr))
        extract_dict['lat'].extend(list(lat_arr))

        if basin_pumping_AF_dir and basin_pumping_mm_dir:     # reading pumping data if directories are provided
            pumping_mm_data = glob(os.path.join(basin_pumping_mm_dir, f'*{year}*.tif'))[0]
            pumping_AF_data = glob(os.path.join(basin_pumping_AF_dir, f'*{year}*.tif'))[0]

            pump_mm_arr = read_raster_arr_object(pumping_mm_data, get_file=False).flatten()
            pump_AF_arr = read_raster_arr_object(pumping_AF_data, get_file=False).flatten()

            extract_dict['pumping_mm'].extend(list(pump_mm_arr))
            extract_dict['pumping_AF'].extend(list(pump_AF_arr))

        else:
            extract_dict['pumping_mm'].extend([None] * len(netGW_arr))
            extract_dict['pumping_AF'].extend([None] * len(netGW_arr))


    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # dropping columns with pumping attribute if the directories were not provided and column contains None
    if not basin_pumping_AF_dir and not basin_pumping_mm_dir:
        df = df[['year', 'netGW_mm', 'lat', 'lon']]

    # converting netGW mm to AF
    area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2
    df['netGW_AF'] = df['netGW_mm'] * area_mm2_single_pixel * 0.000000000000810714  # 1 mm3 = 0.000000000000810714 AF

    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    return output_csv


def compile_pixelwise_netGW_to_annual_df(years, basin_netGW_dir, output_csv):
    """
    Compiling pixel-wise netGW to annual total dataframe.

    :param years: List of years_list to process data.
    :param basin_netGW_dir: Basin netGW directory.
    :param output_csv: Filepath of output csv.

    :return:  Filepath of output csv.
    """
    print(f'Compiling pixel-wise growing season netGW to annual dataframe...')

    # empty dictionary with to store data
    extract_dict = {'year': [], 'netGW_mm': []}

    # lopping through each year and storing data in a list
    for year in years:
        netGW_data = glob(os.path.join(basin_netGW_dir, f'*{year}*.tif'))[0]
        netGW_arr = read_raster_arr_object(netGW_data, get_file=False).flatten()

        year_list = [year] * len(netGW_arr)

        extract_dict['year'].extend(year_list)
        extract_dict['netGW_mm'].extend(list(netGW_arr))

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # converting netGW mm to AF
    area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2
    df['netGW_AF'] = df['netGW_mm'] * area_mm2_single_pixel * 0.000000000000810714  # 1 mm3 = 0.000000000000810714 AF
    df.dropna(inplace=True)

    # summing using groupby()
    df_agg = df.groupby('year')[['netGW_mm', 'netGW_AF']].sum().reset_index()

    df_agg.to_csv(output_csv, index=False)

    return output_csv


def aggregate_USGS_pumping_annual_csv(years, usgs_GW_shp_for_basin, convert_to_crs, output_csv):
    """
    Aggregate USGS HUC12-scale GW pumping estimates to a annual scale for a basin of interest.

    :param years: List of years_list to process the data for.
    :param usgs_GW_shp_for_basin: USGS HUC12-level shapefile (for the basin) with annual GW pumping estimates in AF.
    :param convert_to_crs: For estimating HUC12 areas inside the basin of interest, use this crs.
    :param output_csv: Filepath of output csv with USGS annual GW pumping estimates for a basin of interest.

    :return: Filepath of output csv.
    """
    print(f'aggregating annual USGS GW irrigation vs pumping ...')

    # converting integer years_list to str
    years = [str(y) for y in years]

    # read USGS dataset
    usgs_df = gpd.read_file(usgs_GW_shp_for_basin)

    # estimating area of each huc12 that is inside the basin
    usgs_df = usgs_df.to_crs(convert_to_crs)
    usgs_df['clipped_area_sqkm'] = usgs_df['geometry'].area / (1000 * 1000)  # unit in sqkm

    # using an area filter
    # if clipped_area_sqkm < 30% of areasqkm, removing that from GW_AF calculation
    # if clipped_area_sqkm within 5% of the areasqkm, consider it fully
    # otherwise use the ratio - clipped_area_sqkm/areasqkm to scale GW_AF
    area_ratio = []
    for idx, row in usgs_df.iterrows():
        if row['clipped_area_sqkm'] <= 0.30 * row['areasqkm']:
            area_ratio.append(0)
        elif row['clipped_area_sqkm'] >= 0.95 * row['areasqkm']:
            area_ratio.append(1)
        else:
            area_ratio.append(row['clipped_area_sqkm'] / row['areasqkm'])

    # multiplying by the area ratio to discard little/no coverage huc12s
    usgs_df['area_ratio'] = area_ratio
    usgs_df = usgs_df[years].mul(usgs_df['area_ratio'], axis=0)

    # transposing to bring years_list in a columns
    usgs_df_T = usgs_df.T
    usgs_df_T['year'] = usgs_df_T.index
    usgs_df_T['year'] = usgs_df_T['year'].astype(
        int)  # the 'year' needs to be converted to int for following merging operation
    usgs_df_T = usgs_df_T.reset_index(drop=True)

    usgs_df_T['USGS_AF'] = usgs_df_T.drop('year', axis=1).sum(axis=1)
    usgs_df_T = usgs_df_T[['year', 'USGS_AF']]

    usgs_df_T.to_csv(output_csv, index=False)


def aggregate_netGW_insitu_usgs_pumping_to_annualCSV_KS_CO(pixel_netGW_pumping_csv, annual_usgs_GW_csv,
                                                           area_basin_mm2, output_annual_csv):
    """
    Aggregate (by sum) pixel-wise annual netGW, in-situ pumping records and usgs pumping estimates for a basin
    to a annual csv.     *** used for GMD4, GMD3 in KS and RPB in CO ***

    *** provides annual netGW/in-situ pumping/usgs pumping (in AF/year and mm/year)
    and mean netGW/in-situ pumping/usgs pumping (in mm/year).

    :param pixel_netGW_pumping_csv: Filepath of csv holding pixel-wise annual netGW and pumping data for a basin.
    :param annual_usgs_GW_csv: USGS annual pumping estimates' csv for the basin.
    :param area_basin_mm2: Area of the basin in mm2.
    :param output_annual_csv: Filepath of output annual total/mean netGW/pumping csv.

    :return: None.
    """
    print('Aggregating netGW, in-situ pumping, USGS estimated pumping to a annual csv...')

    # loading dataframe with pixelwise netGW and pumping estimates
    pixel_df = pd.read_csv(pixel_netGW_pumping_csv)

    # groupby using sum()
    yearly_df = pixel_df.groupby('year').sum()
    yearly_df = yearly_df.drop(columns=['lat', 'lon'])
    yearly_df = yearly_df.reset_index()

    # loading USGS annual pumping estimates data
    usgs_df = pd.read_csv(annual_usgs_GW_csv)

    # merging netGW + in-situ pumping + USGS pumping estimates together
    yearly_df = yearly_df.merge(usgs_df, on='year')

    # calculating m3 values
    yearly_df['netGW_m3'] = yearly_df['netGW_AF'] * 1233.48
    yearly_df['pumping_m3'] = yearly_df['pumping_AF'] * 1233.48
    yearly_df['USGS_m3'] = yearly_df['USGS_AF'] * 1233.48

    # calculating mean netGW + mean pumping + mean USGS pumping (in mm)
    yearly_df['mean netGW_mm'] = yearly_df['netGW_AF'] * 1233481837547.5 / area_basin_mm2
    yearly_df['mean pumping_mm'] = yearly_df['pumping_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm
    yearly_df['mean USGS_mm'] = (yearly_df['USGS_AF'] * 1233481837547.5 / area_basin_mm2)  # AF >> mm3 >> mean mm

    # saving final csv
    yearly_df.to_csv(output_annual_csv, index=False)


def aggregate_netGW_insitu_usgs_pumping_to_annualCSV_AZ(pixel_netGW_csv, annual_pumping_csv, basin_code,
                                                        annual_usgs_GW_csv, area_basin_mm2, output_annual_csv):
    """
    Aggregate (by sum) pixel-wise annual netGW, in-situ pumping records and usgs pumping estimates for a basin
    to a annual csv.     *** used for HQR INA and Doug. AMA in AZ  ***

    *** provides annual netGW/in-situ pumping/usgs pumping (in AF/year and mm/year)
    and mean netGW/in-situ pumping/usgs pumping (in mm/year).

    :param pixel_netGW_csv: Filepath of csv holding pixel-wise annual netGW data for a basin.
    :param annual_pumping_csv: In-situ annually summed pumping csv (for all basins in AZ).
    :param annual_usgs_GW_csv: USGS annual pumping estimates' csv for the basin.
    :param basin_code: Either 'hqr' or 'doug' to select from 'HARQUAHALA INA' or 'DOUGLAS AMA'.
    :param area_basin_mm2: Area of the basin in mm2.
    :param output_annual_csv: Filepath of output annual total/mean netGW/pumping csv.

    :return: None.
    """
    print('Aggregating netGW, in-situ pumping, USGS estimated pumping to a annual csv...')

    # loading dataframe with pixelwise netGW estimates
    pixel_df = pd.read_csv(pixel_netGW_csv)
    pixel_df = pixel_df[['year', 'netGW_mm', 'netGW_AF']]

    # groupby using sum()
    netGW_df = pixel_df.groupby('year').sum()
    netGW_df = netGW_df.reset_index()

    # loading annual summed pumping database
    basin_name = 'HARQUAHALA INA' if basin_code == 'hqr' else 'DOUGLAS AMA'
    pump_df = pd.read_csv(annual_pumping_csv)
    pump_df = pump_df[pump_df['AMA INA'] == basin_name]
    pump_df = pump_df[['year', 'AF_sum']]
    pump_df.columns.values[1] = 'pumping_AF'

    # loading USGS annual pumping estimates data
    usgs_df = pd.read_csv(annual_usgs_GW_csv)

    # merging netGW + in-situ pumping + USGS pumping estimates together
    yearly_df = netGW_df.merge(pump_df, on='year')
    yearly_df = yearly_df.merge(usgs_df, on='year')

    # calculating m3 values
    yearly_df['netGW_m3'] = yearly_df['netGW_AF'] * 1233.48
    yearly_df['pumping_m3'] = yearly_df['pumping_AF'] * 1233.48
    yearly_df['USGS_m3'] = yearly_df['USGS_AF'] * 1233.48

    # calculating mean netGW + mean pumping + mean USGS pumping (in mm)
    yearly_df['mean netGW_mm'] = yearly_df['netGW_AF'] * 1233481837547.5 / area_basin_mm2
    yearly_df['mean pumping_mm'] = yearly_df['pumping_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm
    yearly_df['mean USGS_mm'] = yearly_df['USGS_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm

    # saving final csv
    yearly_df.to_csv(output_annual_csv, index=False)


def aggregate_netGW_insitu_usgs_pumping_to_annualCSV_NV_UT(years, basin_netGW_dir,
                                                           pumping_csv, pump_AF_attr,
                                                           annual_usgs_GW_csv,
                                                           area_basin_mm2, output_csv):
    """
    Aggregates (by sum) pixel-wise annual netGW, in-situ pumping records and usgs pumping estimates for
    Diamond Valley, Nevada and Parowan Valley, Utah.

    :param years: List of years_list to process data.
    :param basin_netGW_dir: Basin netGW directory.
    :param pumping_csv: Filepath  of pumping csv dataset for the Diamond valley/Nevada and parowan valley basin.
    :param pump_AF_attr: Pumping attribute (in AF) in the csv file.
    :param annual_usgs_GW_csv: USGS annual pumping estimates' csv for the basin
    :param area_basin_mm2: Area of the basin in mm2.
    :param output_csv: Filepath of output annual total/mean netGW/pumping csv.

    :return: None.
    """
    print(f'Compiling growing season netGW vs annual pumping aggregated dataframe...')

    # empty dictionary with to store data
    extract_dict = {'year': [], 'netGW_mm': []}

    # lopping through each year and storing data in a list
    for year in years:
        netGW_data = glob(os.path.join(basin_netGW_dir, f'*{year}*.tif'))[0]

        netGW_arr = read_raster_arr_object(netGW_data, get_file=False).flatten()
        year_list = [year] * len(netGW_arr)

        extract_dict['year'].extend(year_list)
        extract_dict['netGW_mm'].extend(list(netGW_arr))

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # converting netGW mm to AF
    area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2
    df['netGW_AF'] = df['netGW_mm'] * area_mm2_single_pixel * 0.000000000000810714  # 1 mm3 = 0.000000000000810714 AF

    df = df.dropna().reset_index(drop=True)

    # loading pumping data
    pump_df = pd.read_csv(pumping_csv)

    # aggregating netGW and pumping dataset by years_list
    df_annual = df.groupby('year')[['netGW_AF']].sum().reset_index()
    pump_df_annual = pump_df.groupby('year')[pump_AF_attr].sum().reset_index()

    # joining the annual netGW and in-situ pumping dataframes
    yearly_df = df_annual.merge(pump_df_annual, on='year', how='outer')  # doing outer merge to keep all years_list' records

    # loading USGS annual pumping estimates data
    usgs_df = pd.read_csv(annual_usgs_GW_csv)

    # merging netGW + in-situ pumping + USGS pumping estimates together
    yearly_df = yearly_df.merge(usgs_df, on='year')

    # calculating m3 values
    yearly_df['netGW_m3'] = yearly_df['netGW_AF'] * 1233.48
    yearly_df['pumping_m3'] = yearly_df[pump_AF_attr] * 1233.48  # will get nans in years_list with no pumping records
    yearly_df['USGS_m3'] = yearly_df['USGS_AF'] * 1233.48

    # # calculating mean netGW + mean pumping + mean USGS pumping (in mm)
    yearly_df['mean netGW_mm'] = yearly_df['netGW_AF'] * 1233481837548.5 / area_basin_mm2
    yearly_df['mean pumping_mm'] = yearly_df[pump_AF_attr] * 1233481837548.5 / area_basin_mm2
    yearly_df['mean USGS_mm'] = yearly_df['USGS_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm

    yearly_df.to_csv(output_csv, index=False)


def aggregate_netGW_usgs_pumping_to_annualCSV_CA_ID(annual_netGW_csv, annual_usgs_GW_csv, irr_eff,
                                                    area_basin_mm2, output_annual_csv):
    """
    Aggregate annual netGW and usgs pumping estimates for a basin
    to a annual csv.     *** used for Central Valley, CA and Snake River Basin, ID  ***

    *** provides annual netGW/usgs pumping (in AF/year and mm/year)
    and mean netGW/usgs pumping (in mm/year).

    :param annual_netGW_csv: Filepath of csv holding annual total netGW data for a basin.
    :param annual_usgs_GW_csv: USGS annual pumping estimates' csv for the basin.
    :param irr_eff: Irrigation efficiency factor to estimate pumping from consumptive use.
    :param area_basin_mm2: Area of the basin in mm2.
    :param output_annual_csv: Filepath of output annual total/mean netGW/pumping csv.

    :return: None.
    """
    print('Aggregating annual netGW and USGS estimated pumping to a csv...')

    # loading dataframe with annual netGW estimates
    annual_netGW_df = pd.read_csv(annual_netGW_csv)

    # loading USGS annual pumping estimates data
    usgs_df = pd.read_csv(annual_usgs_GW_csv)

    # merging netGW + USGS pumping estimates together
    yearly_df = annual_netGW_df.merge(usgs_df, on='year')

    # calculating m3 values
    yearly_df['netGW_m3'] = yearly_df['netGW_AF'] * 1233.48
    yearly_df['USGS_m3'] = yearly_df['USGS_AF'] * 1233.48

    # calculating mean netGW + mean USGS pumping (in mm)
    area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2
    yearly_df['mean netGW_mm'] = yearly_df['netGW_AF'] * 1233481837547.5 / area_basin_mm2
    yearly_df['mean USGS_mm'] = yearly_df['USGS_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm

    # estimating pumping from netGW (consumptive use)
    yearly_df['sim_pumping_m3'] = yearly_df['netGW_m3'] / irr_eff
    yearly_df['sim_mean_pumping_mm'] = yearly_df['mean netGW_mm'] / irr_eff

    # saving final csv
    yearly_df.to_csv(output_annual_csv, index=False)


def run_annual_csv_processing_KS_CO(years, basin_code, basin_shp,
                                    westUS_netGW_dir, pumping_pts_shp,
                                    pumping_attr_AF, year_attr,
                                    main_output_dir, pixelwise_output_csv,
                                    usgs_westUS_GW_shp,
                                    usgs_annual_GW_estimates_csv,
                                    final_annual_csv,
                                    skip_processing=False):
    """
    Run processes to compile a basins' netGW, pumping, and USGS pumping data at annual scale in a csv for
    groundwater management districts in KS and republican basin in CO.

    :param years: List of years_list to process data.
    :param basin_code: Basin keyword to get area and save processed datasets. Must be one of the following-
                        ['gmd4', 'gmd3', 'rpb']
    :param basin_shp: Filepath of basin shapefile.
    :param westUS_netGW_dir: WestUS netGW directory.
    :param pumping_pts_shp: Filepath of point shapefile with annual pumping estimates.
    :param pumping_attr_AF: Attribute in the point shapefile with pumping in AF values.
    :param year_attr: Attribute in the point shapefile with year.
    :param main_output_dir: Filepath of main output directory to store processed data for a basin.
    :param pixelwise_output_csv: Filepath of csv holding pixel-wise annual netGW and pumping data for a basin.
    :param usgs_westUS_GW_shp: USGS HUC12-level shapefile (for the Western US) with annual GW pumping estimates in AF.
    :param usgs_annual_GW_estimates_csv: Filepath of output csv with USGS annual GW pumping estimates for the basin.
    :param final_annual_csv: Filepath of final output csv with annual netGW, in-situ pumping, and USGS pumping estimates.
    :param skip_processing: Set to True to skip the processing.

    :return: None.
    """
    if not skip_processing:
        # area of basins
        basin_area_dict = {
            'gmd4': 12737667189.642 * (1000 * 1000),  # in mm2
            'gmd3': 21820149683.491 * (1000 * 1000),  # in mm2
            'rpb': 22753400088.854 * (1000 * 1000),  # in mm2
        }

        # creating output directories for different processes
        # pumping AF and mm raster directories will be created inside the pumping_AF_pts_to_raster() function
        basin_netGW_dir = os.path.join(main_output_dir, 'netGW_basin_mm')
        usgs_basin_GW_dir = os.path.join(main_output_dir, 'USGS_GW_irr')
        usgs_basin_GW_shp = os.path.join(main_output_dir, 'USGS_GW_irr', 'USGS_GW_irr.shp')
        makedirs([basin_netGW_dir, usgs_basin_GW_dir])

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for the basin
        print('# # # # #  STEP 1 # # # # #')

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=basin_shp,
                                      netGW_input_dir=westUS_netGW_dir,
                                      basin_netGW_output_dir=basin_netGW_dir,
                                      resolution=model_res,
                                      irr_frac_input_dir=None,
                                      basin_irr_frac_output_dir=None)

        # # # # #  STEP 2 # # # # #
        # # Converting annual pumping shapefile (unit AF) to mm raster
        print('# # # # #  STEP 2 # # # # #')

        basin_pumping_AF_dir, basin_pumping_mm_dir = \
            pumping_AF_pts_to_raster(years=years, pumping_pts_shp=pumping_pts_shp,
                                     pumping_attr_AF=pumping_attr_AF, year_attr=year_attr,
                                     output_dir=main_output_dir, basin_shp=basin_shp,
                                     ref_raster=WestUS_raster, resolution=model_res)

        # # # # #  STEP 3 # # # # #
        # # Compile pixelwise growing season netGW and annual pumping in dataframes
        print('# # # # #  STEP 3 # # # # #')

        compile_pixelwise_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                                     basin_pumping_mm_dir=basin_pumping_mm_dir,
                                                     basin_pumping_AF_dir=basin_pumping_AF_dir,
                                                     output_csv=pixelwise_output_csv)

        # # # # #  STEP 4 # # # # #
        # # Clip USGS HUC12-scale basins with NHM predicted GW pumping estimates
        print('# # # # #  STEP 4 # # # # #', '\n', 'Clipping HUC12-scale basins with USGS GW pumping data...')

        clip_vector(input_shapefile=usgs_westUS_GW_shp, mask_shapefile=basin_shp,
                    output_shapefile=usgs_basin_GW_shp, create_zero_buffer=False,
                    change_crs='EPSG:4269')  # the conversion to EPSG 4269 is needed as all basin shapefiles are in this crs

        # # # # #  STEP 5 # # # # #
        # # Aggregate USGS HUC12-scale GW pumping estimates to a annual scale for the basin of interest
        print('# # # # #  STEP 5 # # # # #')

        aggregate_USGS_pumping_annual_csv(years=years, usgs_GW_shp_for_basin=usgs_basin_GW_shp,
                                          convert_to_crs='EPSG:3857',
                                          output_csv=usgs_annual_GW_estimates_csv)

        # # # # #  STEP 6 # # # # #
        # # Compile the basin's pixelwise netGW and in-situ pumping to a common csv
        print('# # # # #  STEP 6 # # # # #')

        aggregate_netGW_insitu_usgs_pumping_to_annualCSV_KS_CO(pixel_netGW_pumping_csv=pixelwise_output_csv,
                                                               annual_usgs_GW_csv=usgs_annual_GW_estimates_csv,
                                                               area_basin_mm2=basin_area_dict[basin_code],
                                                               output_annual_csv=final_annual_csv)
    else:
        pass


def run_annual_csv_processing_AZ(years, basin_code, basin_shp,
                                 westUS_netGW_dir, annual_pumping_csv,
                                 main_output_dir, pixelwise_output_csv,
                                 usgs_westUS_GW_shp,
                                 usgs_annual_GW_estimates_csv,
                                 final_annual_csv,
                                 skip_processing=False):
    """
    Run processes to compile a basins' netGW, pumping, and USGS pumping data at annual scale in a csv for
    Harquahala INA and Douglas AMA in Arizona.

    :param years: List of years_list to process data.
    :param basin_code: Basin keyword to get area and save processed datasets. Must be one of the following-
                        ['hqr', 'doug']
    :param basin_shp: Filepath of basin shapefile.
    :param westUS_netGW_dir: WestUS netGW directory.
    :param annual_pumping_csv: Filepath of annual basin aggregated pumping database (csv).
    :param main_output_dir: Filepath of main output directory to store processed data for a basin.
    :param pixelwise_output_csv: Filepath of csv holding pixel-wise annual netGW and pumping data for a basin.
    :param usgs_westUS_GW_shp: USGS HUC12-level shapefile (for the Western US) with annual GW pumping estimates in AF.
    :param usgs_annual_GW_estimates_csv: Filepath of output csv with USGS annual GW pumping estimates for the basin.
    :param final_annual_csv: Filepath of final output csv with annual netGW, in-situ pumping, and USGS pumping estimates.
    :param skip_processing: Set to True to skip the processing.

    :return: None.
    """
    if not skip_processing:
        # area of basins
        basin_area_dict = {
            'hqr': 1982641859.510 * (1000 * 1000),  # in mm2
            'doug': 2459122191.981 * (1000 * 1000),  # in mm2
        }

        # creating output directories for different processes
        # pumping AF and mm raster directories will be created inside the pumping_AF_pts_to_raster() function
        basin_netGW_dir = os.path.join(main_output_dir, 'netGW_basin_mm')
        usgs_basin_GW_dir = os.path.join(main_output_dir, 'USGS_GW_irr')
        usgs_basin_GW_shp = os.path.join(main_output_dir, 'USGS_GW_irr', 'USGS_GW_irr.shp')
        makedirs([basin_netGW_dir, usgs_basin_GW_dir])

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for the basin
        print('# # # # #  STEP 1 # # # # #')

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=basin_shp,
                                      netGW_input_dir=westUS_netGW_dir,
                                      basin_netGW_output_dir=basin_netGW_dir,
                                      resolution=model_res,
                                      irr_frac_input_dir=None,
                                      basin_irr_frac_output_dir=None)

        # # # # #  STEP 2 # # # # #
        # # Compile pixel-wise growing season netGW and annual pumping in dataframes
        print('# # # # #  STEP 2 # # # # #')

        compile_pixelwise_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                                     basin_pumping_mm_dir=None,
                                                     basin_pumping_AF_dir=None,
                                                     output_csv=pixelwise_output_csv)

        # # # # #  STEP 3 # # # # #
        # # Clip USGS HUC12-scale basins with NHM predicted GW pumping estimates
        print('# # # # #  STEP 3 # # # # #', '\n', 'Clipping HUC12-scale basins with USGS GW pumping data...')

        clip_vector(input_shapefile=usgs_westUS_GW_shp, mask_shapefile=basin_shp,
                    output_shapefile=usgs_basin_GW_shp, create_zero_buffer=False,
                    change_crs='EPSG:4269')  # the conversion to EPSG 4269 is needed as all basin shapefiles are in this crs

        # # # # #  STEP 4 # # # # #
        # # Aggregate USGS HUC12-scale GW pumping estimates to a annual scale for the basin of interest
        print('# # # # #  STEP 4 # # # # #')

        aggregate_USGS_pumping_annual_csv(years=years, usgs_GW_shp_for_basin=usgs_basin_GW_shp,
                                          convert_to_crs='EPSG:3857',
                                          output_csv=usgs_annual_GW_estimates_csv)

        # # # # #  STEP 5 # # # # #
        # # Compile the basin's pixelwise netGW and in-situ pumping to a common csv
        print('# # # # #  STEP 5 # # # # #')

        aggregate_netGW_insitu_usgs_pumping_to_annualCSV_AZ(pixel_netGW_csv=pixelwise_output_csv,
                                                            annual_pumping_csv=annual_pumping_csv,
                                                            basin_code=basin_code,
                                                            annual_usgs_GW_csv=usgs_annual_GW_estimates_csv,
                                                            area_basin_mm2=basin_area_dict[basin_code],
                                                            output_annual_csv=final_annual_csv)
    else:
        pass


def run_annual_csv_processing_NV_UT(years, basin_code, basin_shp,
                                    westUS_netGW_dir,
                                    pumping_csv, pumping_attr_AF,
                                    main_output_dir,
                                    usgs_westUS_GW_shp,
                                    usgs_annual_GW_estimates_csv,
                                    final_annual_csv,
                                    skip_processing=False):
    """
    Run processes to compile Diamond Valley's (Nevada) and Parowan Valley's (Utah) netGW, pumping, and USGS pumping data
    at annual scale in a csv.

    :param years: List of years_list to process data.
    :param basin_code: Basin keyword to get area and save processed datasets. Must be - 'dv'.
    :param basin_shp: Filepath of basin shapefile.
    :param westUS_netGW_dir: WestUS netGW directory.
    :param pumping_csv: Filepath of in-situ pumping data csv.
    :param pumping_attr_AF: Attribute in the point shapefile with pumping in AF values.
    :param main_output_dir: Filepath of main output directory to store processed data for a basin.
    :param usgs_westUS_GW_shp: USGS HUC12-level shapefile (for the Western US) with annual GW pumping estimates in AF.
    :param usgs_annual_GW_estimates_csv: Filepath of output csv with USGS annual GW pumping estimates for the basin.
    :param final_annual_csv: Filepath of final output csv with annual netGW, in-situ pumping, and USGS pumping estimates.
    :param skip_processing: Set to True to skip the processing.

    :return: None.
    """
    if not skip_processing:
        # area of basins
        basin_area_dict = {'dv': 1933578136.225 * (1000 * 1000), 'pv': 1339578824.848 * (1000 * 1000)}  # in mm2

        # creating output directories for different processes
        basin_netGW_dir = os.path.join(main_output_dir, 'netGW_basin_mm')
        usgs_basin_GW_dir = os.path.join(main_output_dir, 'USGS_GW_irr')
        usgs_basin_GW_shp = os.path.join(main_output_dir, 'USGS_GW_irr', 'USGS_GW_irr.shp')
        makedirs([basin_netGW_dir, usgs_basin_GW_dir])

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for the basin
        print('# # # # #  STEP 1 # # # # #')

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=basin_shp,
                                      netGW_input_dir=westUS_netGW_dir,
                                      basin_netGW_output_dir=basin_netGW_dir,
                                      resolution=model_res,
                                      irr_frac_input_dir=None,
                                      basin_irr_frac_output_dir=None)

        # # # # #  STEP 2 # # # # #
        # # Clip USGS HUC12-scale basins with NHM predicted GW pumping estimates
        print('# # # # #  STEP 2 # # # # #', '\n', 'Clipping HUC12-scale basins with USGS GW pumping data...')

        clip_vector(input_shapefile=usgs_westUS_GW_shp, mask_shapefile=basin_shp,
                    output_shapefile=usgs_basin_GW_shp, create_zero_buffer=False,
                    change_crs='EPSG:4269')  # the conversion to EPSG 4269 is needed as all basin shapefiles are in this crs

        # # # # #  STEP 3 # # # # #
        # # Aggregate USGS HUC12-scale GW pumping estimates to a annual scale for the basin of interest
        print('# # # # #  STEP 3 # # # # #')

        aggregate_USGS_pumping_annual_csv(years=years, usgs_GW_shp_for_basin=usgs_basin_GW_shp,
                                          convert_to_crs='EPSG:3857',
                                          output_csv=usgs_annual_GW_estimates_csv)

        # # # # #  STEP 4 # # # # #
        # # Compile the basin's pixelwise netGW and in-situ pumping to a common csv
        print('# # # # #  STEP 4 # # # # #')

        aggregate_netGW_insitu_usgs_pumping_to_annualCSV_NV_UT(years=years, basin_netGW_dir=basin_netGW_dir,
                                                               pumping_csv=pumping_csv, pump_AF_attr=pumping_attr_AF,
                                                               annual_usgs_GW_csv=usgs_annual_GW_estimates_csv,
                                                               area_basin_mm2=basin_area_dict[basin_code],
                                                               output_csv=final_annual_csv)
    else:
        pass


def run_annual_csv_processing_CA_ID(years, basin_code, basin_shp, westUS_netGW_dir,
                                    main_output_dir, annual_netGW_output_csv,
                                    usgs_westUS_GW_shp,
                                    usgs_annual_GW_estimates_csv,
                                    irr_eff, final_annual_csv,
                                    skip_processing=False):
    """
    Run processes to compile a basins' netGW, pumping, and USGS pumping data at annual scale in a csv for
    Central Valley, CA and Snake River Basin, ID.

    :param years: List of years_list to process data.
    :param basin_code: Basin keyword to get area and save processed datasets. Must be one of the following-
                        ['cv', 'srb]
    :param basin_shp: Filepath of basin shapefile.
    :param westUS_netGW_dir: WestUS netGW directory.
    :param main_output_dir: Filepath of main output directory to store processed data for a basin.
    :param annual_netGW_output_csv: Filepath of csv holding total annual netGW data for the basin.
    :param usgs_westUS_GW_shp: USGS HUC12-level shapefile (for the Western US) with annual GW pumping estimates in AF.
    :param usgs_annual_GW_estimates_csv: Filepath of output csv with USGS annual GW pumping estimates for the basin.
    :param irr_eff: Irrigation efficiency factor to estimate pumping from consumptive use.
    :param final_annual_csv: Filepath of final output csv with annual netGW, in-situ pumping, and USGS pumping estimates.
    :param skip_processing: Set to True to skip the processing.

    :return: None.
    """
    if not skip_processing:
        # area of basins
        basin_area_dict = {
            'cv': 52592338000 * (1000 * 1000),  # in mm2
            'srb': 39974555000 * (1000 * 1000),  # in mm2

        }

        # creating output directories for different processes
        basin_netGW_dir = os.path.join(main_output_dir, 'netGW_basin_mm')
        usgs_basin_GW_dir = os.path.join(main_output_dir, 'USGS_GW_irr')
        usgs_basin_GW_shp = os.path.join(main_output_dir, 'USGS_GW_irr', 'USGS_GW_irr.shp')
        makedirs([basin_netGW_dir, usgs_basin_GW_dir])

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for the basin
        print('# # # # #  STEP 1 # # # # #')

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=basin_shp,
                                      netGW_input_dir=westUS_netGW_dir,
                                      basin_netGW_output_dir=basin_netGW_dir,
                                      resolution=model_res,
                                      irr_frac_input_dir=None,
                                      basin_irr_frac_output_dir=None)

        # # # # #  STEP 2 # # # # #
        # # Compile pixelwise growing season netGW to annual total dataframe
        print('# # # # #  STEP 2 # # # # #')
        compile_pixelwise_netGW_to_annual_df(years=years, basin_netGW_dir=basin_netGW_dir,
                                             output_csv=annual_netGW_output_csv)

        # # # # #  STEP 3 # # # # #
        # # Clip USGS HUC12-scale basins with NHM predicted GW pumping estimates
        print('# # # # #  STEP 3 # # # # #', '\n', 'Clipping HUC12-scale basins with USGS GW pumping data...')

        clip_vector(input_shapefile=usgs_westUS_GW_shp, mask_shapefile=basin_shp,
                    output_shapefile=usgs_basin_GW_shp, create_zero_buffer=False,
                    change_crs='EPSG:4269')  # the conversion to EPSG 4269 is needed as all basin shapefiles are in this crs

        # # # # #  STEP 4 # # # # #
        # # Aggregate USGS HUC12-scale GW pumping estimates to a annual scale for the basin of interest
        print('# # # # #  STEP 4 # # # # #')

        aggregate_USGS_pumping_annual_csv(years=years, usgs_GW_shp_for_basin=usgs_basin_GW_shp,
                                          convert_to_crs='EPSG:3857',
                                          output_csv=usgs_annual_GW_estimates_csv)

        # # # # #  STEP 5 # # # # #
        # # Compile the basin's total annual netGW and USGS pumping to a common csv
        print('# # # # #  STEP 5 # # # # #')

        aggregate_netGW_usgs_pumping_to_annualCSV_CA_ID(annual_netGW_csv=annual_netGW_output_csv,
                                                        annual_usgs_GW_csv=usgs_annual_GW_estimates_csv,
                                                        irr_eff=irr_eff,
                                                        area_basin_mm2=basin_area_dict[basin_code],
                                                        output_annual_csv=final_annual_csv)

    else:
        pass


def compile_annual_pumping_netGW_all_basins(annual_csv_list, output_csv):
    """
    Compile all basins' annual netGW, in-situ pumping, USGS pumping estimates in a single csv.

    :param annual_csv_list: List of basins' annual csv filepaths.
    :param output_csv: Filepath of compiled annual output csv.

    :return: None.
    """
    # making the output directory if not available
    makedirs([os.path.dirname(output_csv)])

    # empty dataframe to store the results
    compiled_annual_df = pd.DataFrame()

    # basin name dict
    basin_name_dict = {'gmd4': 'GMD4, KS', 'gmd3': 'GMD3, KS', 'rpb': 'Republican Basin, CO',
                       'hqr': 'Harquahala INA, AZ', 'doug': 'Douglas AMA, AZ',
                       'dv': 'Diamond Valley, NV', 'pv': 'Parowan Valley, UT'}

    for csv in annual_csv_list:
        df = pd.read_csv(csv)

        basin_name = os.path.basename(csv).split('_')[0]
        df['basin_code'] = basin_name
        df['basin'] = [basin_name_dict[basin_name] for i in range(len(df))]
        compiled_annual_df = pd.concat([compiled_annual_df, df])

    # creating dynamic irrigation efficiency dictionary for the basins
    basin_irr_eff_dict = {'gmd4': 0.88, 'gmd3': 0.88, 'rpb': 0.80,
                          'hqr': 0.75, 'doug': 0.85, 'dv': 0.87, 'pv': 0.78}

    # estimating volumetric and mean pumping based on basin-specific efficiency
    compiled_annual_df['dyn_irr_eff'] = compiled_annual_df.apply(lambda x: basin_irr_eff_dict[x['basin_code']], axis=1)
    compiled_annual_df['sim_pumping_m3_dy'] = compiled_annual_df['netGW_m3'] / compiled_annual_df['dyn_irr_eff']
    compiled_annual_df['sim_mean_pumping_mm_dy'] = compiled_annual_df['mean netGW_mm'] / compiled_annual_df['dyn_irr_eff']

    # Error range with 70-90% efficiency
    compiled_annual_df['sim_pumping_m3_70'] = compiled_annual_df['netGW_m3'] / 0.70
    compiled_annual_df['sim_pumping_m3_90'] = compiled_annual_df['netGW_m3'] / 0.90

    compiled_annual_df['sim_mean_pumping_mm_70'] = compiled_annual_df['mean netGW_mm'] / 0.70
    compiled_annual_df['sim_mean_pumping_mm_90'] = compiled_annual_df['mean netGW_mm'] / 0.90

    # setting all zero values to np.nan
    compiled_annual_df = compiled_annual_df.replace({0: np.nan})

    compiled_annual_df.to_csv(output_csv, index=False)


def extract_pumping_estimate_with_lat_lon(years, input_csv, input_data_dir, resampled_output_dir,
                                          output_csv, ref_raster=WestUS_raster,
                                          resolution=model_res):
    """
    Extract pumping values using latitude and longitude from a pumping value raster.
    ** using it to extract pumping value from Majumdar et al.'s simulated Arizona's pumping data.

    :param years: List of years_list to process data.
    :param input_csv: Filepath of original pixel-wise annual netGW and pumping csv for a basin.
    :param input_data_dir: Input pumping data directory path.
    :param resampled_output_dir: Directory path to save intermediate resampled pumping data.
    :param output_csv: Filepath of out csv that holds extracted pumping values.
    :param ref_raster: Filepath of Western US reference raster.
    :param resolution: model resolution.

    :return: None
    """
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
        resampled_pumping_data = \
            clip_resample_reproject_raster(input_raster=pumping_data, input_shape=None,
                                           output_raster_dir=resampled_output_dir,
                                           keyword=' ', raster_name=raster_name,
                                           clip=False, resample=True, clip_and_resample=False,
                                           targetaligned=True, resample_algorithm='near',
                                           resolution=resolution,
                                           crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                           ref_raster=ref_raster, use_ref_width_height=True)

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


def process_and_aggregate_irrigated_acres(years, irr_cropland_input_dir, irr_frac_input_dir, basin_shp, basin_name,
                                          in_situ_areacsv, main_output_dir, output_csv, resolution=model_res,
                                          in_situ_shp_list_CO=None):
    # creating output dirs
    global insitu_df

    basin_irr_frac_output_dir = os.path.join(main_output_dir, 'irr_frac')
    basin_irr_cropland_output_dir = os.path.join(main_output_dir, 'irr_cropland')

    makedirs([basin_irr_frac_output_dir, basin_irr_cropland_output_dir])

    # empty dictionary to store results
    acre_dict = {'Year': [], 'Irr data Acres': []}

    for year in years:
        print(f'Clipping irrigated cropland and fraction data for {year}...')

        # irrigation fraction
        irr_frac_raster = glob(os.path.join(irr_frac_input_dir, f'*{year}*.tif'))[0]

        basin_irr_frac_data = clip_resample_reproject_raster(input_raster=irr_frac_raster, input_shape=basin_shp,
                                                             output_raster_dir=basin_irr_frac_output_dir,
                                                             keyword=' ', raster_name=f'Irr_frac_{year}.tif',
                                                             clip=True, resample=False, clip_and_resample=False,
                                                             targetaligned=True, resample_algorithm='near',
                                                             resolution=resolution,
                                                             crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                                             use_ref_width_height=False)

        # irrigation cropland
        irr_crop_raster = glob(os.path.join(irr_cropland_input_dir, f'*{year}*.tif'))[0]

        basin_irr_cropland_data = clip_resample_reproject_raster(input_raster=irr_crop_raster, input_shape=basin_shp,
                                                                 output_raster_dir=basin_irr_cropland_output_dir,
                                                                 keyword=' ', raster_name=f'Irr_cropland_{year}.tif',
                                                                 clip=True, resample=False, clip_and_resample=False,
                                                                 targetaligned=True, resample_algorithm='near',
                                                                 resolution=resolution,
                                                                 crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                                                 use_ref_width_height=False)

        # calculating irrigated area
        irr_frac_arr, file = read_raster_arr_object(basin_irr_frac_data)
        irr_cropland_arr = read_raster_arr_object(basin_irr_cropland_data, get_file=False)

        area_acres_single_pixel = (2.193 ** 2) * 247.105  # unit in acres
        irr_acreage_arr = np.where((irr_frac_arr != -9999), irr_frac_arr * irr_cropland_arr * area_acres_single_pixel,
                                   -9999)

        total_irr_acreage = np.nansum(irr_acreage_arr)

        # appending result to dictionary
        acre_dict['Year'].append(year)
        acre_dict['Irr data Acres'].append(total_irr_acreage)

    # converting dictionary to dataframe
    irr_acre_df = pd.DataFrame(acre_dict)

    # loading and summing (annually) in-situ area data
    if in_situ_areacsv is not None:  # only required for GMD3, GMD4 in KS and Diamond Valley, CO
        insitu_df = pd.read_csv(in_situ_areacsv)

    # for Kansas basins
    if basin_name in ['gmd3', 'gmd4']:
        basin_code = 3 if basin_name == 'gmd3' else 4  # we are only considering gmd4 or gmd3 for this project. The database has all gmds' data

        insitu_df = insitu_df[insitu_df['gmd'] == basin_code]
        df_agg = insitu_df.groupby('Year')['Acres'].sum().reset_index()
        df_agg = df_agg.rename(columns={'Acres': 'In situ Acres'})

        # compiling in-situ irrigated area records and irrigated area from irrigated datasets
        df_agg = df_agg.merge(irr_acre_df, on='Year')

        # converting acres to sq meter (SI unit)
        df_agg['In situ m2'] = df_agg['In situ Acres'] * 4046.8564224  # sq meter
        df_agg['Irr data m2'] = df_agg['Irr data Acres'] * 4046.8564224  # sq meter

    # for Nevada basins
    elif basin_name == 'dv':
        insitu_df = insitu_df.groupby('year')['area_m2'].sum().reset_index()
        insitu_df = insitu_df.rename(columns={'year': 'Year', 'area_m2': 'In situ m2'})

        # compiling in-situ irrigated area records and irrigated area from irrigated datasets
        df_agg = insitu_df.merge(irr_acre_df, on='Year')

        # converting sq meter (SI unit) to Acres and vice versa
        df_agg['In situ Acres'] = df_agg['In situ m2'] / 4046.8564224  # sq meter
        df_agg['Irr data m2'] = df_agg['Irr data Acres'] * 4046.8564224  # sq meter

    # for Colorado basins
    elif (basin_name == 'rpb') & (in_situ_shp_list_CO is not None):
        # loading and summing (annually) in-situ acreage data
        in_situ_dict = {'Year': [], 'In situ m2': []}

        for shp in in_situ_shp_list_CO:  # looping through each year's shapefile and estimating acres
            gdf = gpd.read_file(shp)

            year = os.path.basename(shp).split('.')[0][-4:]
            area = gdf.geometry.area.sum()  # unit m2

            # appending result to dictionary
            in_situ_dict['Year'].append(int(year))
            in_situ_dict['In situ m2'].append(area)

        # converting dictionary to dataframe
        in_situ_df = pd.DataFrame(in_situ_dict)

        # compiling in-situ irrigated acreage records and irrigated acreage from irrigated datasets
        df_agg = irr_acre_df.merge(in_situ_df, on='Year')

        # converting acres to sq meter (SI unit) and sq meter to acres
        df_agg['Irr data m2'] = df_agg['Irr data Acres'] * 4046.8564224  # sq meter
        df_agg['In situ Acres'] = df_agg['In situ m2'] / 4046.8564224  # Acres

    else:
        print('Check input arguments')

    # assigning basin name
    basin_name_dict = {'gmd3': 'GMD3, KS', 'gmd4': 'GMD4, KS', 'dv': 'Diamond Valley, NV',
                       'rpb': 'Republican Basin, CO'}
    basin_name = basin_name_dict[basin_name]
    df_agg['basin_name'] = [basin_name] * len(df_agg)

    # re-arranging columns as KS data
    df_agg = df_agg[['Year', 'In situ Acres', 'Irr data Acres', 'In situ m2', 'Irr data m2', 'basin_name']]

    # save
    df_agg.to_csv(output_csv, index=False)


def compile_irr_acres_all_basins(annual_csv_list, output_csv):
    # making the output directory if not available
    makedirs([os.path.dirname(output_csv)])

    # empty dataframe to store the results
    compiled_annual_df = pd.DataFrame()

    for csv in annual_csv_list:
        df = pd.read_csv(csv)
        compiled_annual_df = pd.concat([compiled_annual_df, df])

    compiled_annual_df.to_csv(output_csv, index=False)


def compile_annual_irr_rainfed_ET(years, area_code, area_shape, area_ref_raster,
                                  cdl_input_dir, irrigated_cropland_input_dir,
                                  rainfed_cropland_input_dir, irrigated_cropET_input_dir,
                                  rainfed_cropET_input_dir,
                                  output_csv,
                                  skip_processing=False, resolution=model_res):
    """
    Compile annual irrigated + rainfed cropland, annual cdl, and monthly (growing season months)
    irrigated + rainfed cropET dataset into a csv.

    :param years: List of years_list' data to include in the dataframe.
    :param area_code: A shortname of area/state that will be used to save the data, e.g., 'KS' or 'TX'.
    :param area_shape: Filepath of the shapefile of the area.
    :param area_ref_raster: Filepath of the refraster of the area.
    :param cdl_input_dir: Filepath of cdl input dir.
    :param irrigated_cropland_input_dir: Filepath of irrigated cropland input dir.
    :param rainfed_cropland_input_dir: Filepath of rainfed cropland input dir.
    :param irrigated_cropET_input_dir: Filepath of monthly irrigated cropET input dir.
    :param rainfed_cropET_input_dir: Filepath of monthly rainfed cropET input dir.
    :param output_csv: Filepath of output annual csv.
    :param skip_processing: Set to True to skip this step..
    :param resolution: Target resolution. Default set to model resolution.

    :return: None.
    """
    if not skip_processing:
        print(f'processing datasets + annual csv for {area_code}...')

        # creating output directories
        USDA_CDL_output_dir = f'../../Data_main/results_eval/rainfed_cropET_compare/{area_code}/USDA_CDL'
        irrigated_cropland_output_dir = f'../../Data_main/results_eval/rainfed_cropET_compare/{area_code}/Irrigated_cropland'
        rainfed_cropland_output_dir = f'../../Data_main/results_eval/rainfed_cropET_compare/{area_code}//Rainfed_cropland'
        irrigated_cropET_output_dir = f'../../Data_main/results_eval/rainfed_cropET_compare/{area_code}/Irrigated_cropET/monthly'
        rainfed_cropET_output_dir = f'../../Data_main/results_eval/rainfed_cropET_compare/{area_code}/Rainfed_cropET/monthly'

        makedirs([USDA_CDL_output_dir, irrigated_cropland_output_dir, rainfed_cropland_output_dir,
                  irrigated_cropET_output_dir, rainfed_cropET_output_dir])

        # clipping + resampling of cdl data
        cdl_datasets = glob(os.path.join(cdl_input_dir, '*.tif'))

        for cdl in cdl_datasets:
            if any(str(i) in cdl for i in years):
                clip_resample_reproject_raster(input_raster=cdl, input_shape=area_shape,
                                               output_raster_dir=USDA_CDL_output_dir,
                                               raster_name=None, clip_and_resample=True,
                                               resolution=resolution, ref_raster=area_ref_raster)

        # clipping + resampling of irrigated cropland data
        irrig_cropland_datasets = glob(os.path.join(irrigated_cropland_input_dir, '*.tif'))
        clipped_output_dir = os.path.join(irrigated_cropland_output_dir, 'interim')

        for cropland in irrig_cropland_datasets:
            if any(str(i) in cropland for i in years):
                clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=area_shape,
                                                                output_raster_dir=clipped_output_dir,
                                                                raster_name=None, clip_and_resample=True,
                                                                resolution=resolution, ref_raster=area_ref_raster)

                # replacing nan values where reference raster is zero with zero
                ref_arr, ref_file = read_raster_arr_object(area_ref_raster)
                clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

                clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

                output_raster = os.path.join(irrigated_cropland_output_dir, os.path.basename(clipped_raster))
                write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                      output_path=output_raster)

        # clipping + resampling of rainfed cropland data
        rain_cropland_datasets = glob(os.path.join(rainfed_cropland_input_dir, '*.tif'))
        clipped_output_dir = os.path.join(rainfed_cropland_output_dir, 'interim')

        for cropland in rain_cropland_datasets:
            if any(str(i) in cropland for i in years):
                clipped_raster = clip_resample_reproject_raster(input_raster=cropland, input_shape=area_shape,
                                                                output_raster_dir=clipped_output_dir,
                                                                raster_name=None, clip_and_resample=True,
                                                                resolution=resolution, ref_raster=area_ref_raster)

                # replacing nan values where reference raster is zero with zero
                ref_arr, ref_file = read_raster_arr_object(area_ref_raster)
                clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

                clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

                output_raster = os.path.join(rainfed_cropland_output_dir, os.path.basename(clipped_raster))
                write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                      output_path=output_raster)

        # clipping + resampling of irrigated cropET data
        irrig_cropET_grow_season_datasets = glob(os.path.join(irrigated_cropET_input_dir, '*.tif'))
        clipped_output_dir = os.path.join(irrigated_cropET_output_dir, 'interim')

        for cropET in irrig_cropET_grow_season_datasets:
            clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=area_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=resolution, ref_raster=area_ref_raster)

            # replacing nan values where reference raster is zero with zero
            ref_arr, ref_file = read_raster_arr_object(area_ref_raster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(irrigated_cropET_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

        # clipping + resampling of rainfed cropET data
        rain_cropET_grow_season_datasets = glob(os.path.join(rainfed_cropET_input_dir, '*.tif'))
        clipped_output_dir = os.path.join(rainfed_cropET_output_dir, 'interim')
        for cropET in rain_cropET_grow_season_datasets:
            clipped_raster = clip_resample_reproject_raster(input_raster=cropET, input_shape=area_shape,
                                                            output_raster_dir=clipped_output_dir,
                                                            raster_name=None, clip_and_resample=True,
                                                            resolution=resolution, ref_raster=area_ref_raster)

            # replacing nan values where reference raster is zero with zero
            ref_arr, ref_file = read_raster_arr_object(area_ref_raster)
            clipped_arr = read_raster_arr_object(clipped_raster, get_file=False)

            clipped_arr = np.where(np.isnan(clipped_arr) & (ref_arr == 0), ref_arr, clipped_arr)

            output_raster = os.path.join(rainfed_cropET_output_dir, os.path.basename(clipped_raster))
            write_array_to_raster(raster_arr=clipped_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_raster)

        # Compiling annual data to CSV
        monthly_data_path_dict = {'Irrigated_cropET': irrigated_cropET_output_dir,
                                  'Rainfed_cropET': rainfed_cropET_output_dir}

        yearly_data_path_dict = {'USDA_CDL': USDA_CDL_output_dir,
                                 'Irrigated': irrigated_cropland_output_dir,
                                 'Rainfed': rainfed_cropland_output_dir}

        datasets_to_include = ['Irrigated_cropET', 'Rainfed_cropET', 'USDA_CDL', 'Irrigated', 'Rainfed']

        makedirs([os.path.dirname(output_csv)])

        create_train_test_monthly_dataframe(years_list=years,
                                            monthly_data_path_dict=monthly_data_path_dict,
                                            yearly_data_path_dict=yearly_data_path_dict,
                                            static_data_path_dict=None,
                                            datasets_to_include=datasets_to_include,
                                            output_parquet=output_csv,
                                            skip_processing=False)


def clip_Peff_for_basin(years, basin_shp, Peff_input_dir, basin_Peff_output_dir, basin_code,
                        month_range=None, resolution=model_res):
    """
    Clip effective precipitation estimates for growing seasons or months.

    :param years: List of years_list to process data.
    :param basin_shp: Filepath of basin shapefile.
    :param Peff_input_dir: Directory path of Western US effective precipitation estimates (growing seasons or months).
    :param basin_Peff_output_dir: Output directory path to save the clipped effective precipitation estimates for the basin.
    :param basin_code: Basin keyword to add before the processed raster's name. Preferred to use from
                          ['gmd3', 'gmd4', 'rpb', 'hqr', 'doug', 'dv', 'cv'].
    :param month_range: Range of month to process data for. Example - (4, 10).
                        Default set to None to process growing season effective precipitation data only.
    :param resolution: model resolution.

    :return: None.
    """
    makedirs([basin_Peff_output_dir])

    for year in years:
        if month_range is None:
            print(f'Clipping effective precipitation for {year}...')

            peff_raster = glob(os.path.join(Peff_input_dir, f'*{year}*.tif'))[0]

            clip_resample_reproject_raster(input_raster=peff_raster, input_shape=basin_shp,
                                           output_raster_dir=basin_Peff_output_dir,
                                           keyword=basin_code, raster_name=None,
                                           clip=True, resample=False, clip_and_resample=False,
                                           targetaligned=True, resample_algorithm='near',
                                           resolution=resolution,
                                           crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                           use_ref_width_height=False)

        else:  # for monthly effective precipitation estimates
            months = list(range(month_range[0], month_range[1] + 1))

            for month in months:
                print(f'Clipping effective precipitation for {year=}, {month=} ...')

                peff_raster = glob(os.path.join(Peff_input_dir, f'*{year}_{month}*.tif'))[0]

                clip_resample_reproject_raster(input_raster=peff_raster, input_shape=basin_shp,
                                               output_raster_dir=basin_Peff_output_dir,
                                               keyword=' ', raster_name=None,
                                               clip=True, resample=False, clip_and_resample=False,
                                               targetaligned=True, resample_algorithm='near',
                                               resolution=resolution,
                                               crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                               use_ref_width_height=False)


def clip_precip_for_basin(years, basin_shp, precip_input_dir, basin_precip_output_dir,
                          basin_code, month_range=None, resolution=model_res):
    """
    Clip effective precipitation estimates for growing seasons or months for the basin.

    :param years: List of years_list to process data.
    :param basin_shp: Filepath of basin shapefile.
    :param precip_input_dir: Directory path of Western US water year precipitation estimates.
    :param basin_precip_output_dir: Output directory path to save the clipped water year precipitation estimates for the basin.
    :param basin_code: Basin keyword to add before the processed raster's name. Preferred to use from
                      ['gmd3', 'gmd4', 'rpb', 'hqr', 'doug', 'dv', 'cv'].

    :param resolution: model resolution.

    :return: None.
    """
    makedirs([basin_precip_output_dir])

    for year in years:
        if month_range is None:
            print(f'Clipping water year precipitation for {year}...')

            precip_raster = glob(os.path.join(precip_input_dir, f'*{year}*.tif'))[0]

            clip_resample_reproject_raster(input_raster=precip_raster, input_shape=basin_shp,
                                           output_raster_dir=basin_precip_output_dir,
                                           keyword=basin_code, raster_name=None,
                                           clip=True, resample=False, clip_and_resample=False,
                                           targetaligned=True, resample_algorithm='near',
                                           resolution=resolution,
                                           crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                           use_ref_width_height=False)
        else:  # for monthly precipitation estimates
            months = list(range(month_range[0], month_range[1] + 1))

            for month in months:
                print(f'Clipping monthly precipitation for {year=}, {month=} ...')

                precip_raster = glob(os.path.join(precip_input_dir, f'*{year}_{month}*.tif'))[0]

                clip_resample_reproject_raster(input_raster=precip_raster, input_shape=basin_shp,
                                               output_raster_dir=basin_precip_output_dir,
                                               keyword=' ', raster_name=None,
                                               clip=True, resample=False, clip_and_resample=False,
                                               targetaligned=True, resample_algorithm='near',
                                               resolution=resolution,
                                               crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                               use_ref_width_height=False)


def compile_basin_growS_peff_water_yr_precip_to_csv(years, basin_peff_dir, basin_water_yr_precip_dir,
                                                    basin_code, output_csv):
    """
    Compiling pixel-wise growing season effective precipitation and water year precipitation data for a basin into a csv.

    :param years: List of years_list to process data.
    :param basin_peff_dir: Basin effective precipitation directory.
    :param basin_water_yr_precip_dir: Basin water year precipitation directory.
    :param basin_code: Basin keyword to add before the processed raster's name. Preferred to use from
                  ['gmd3', 'gmd4', 'rpb', 'hqr', 'doug', 'dv', 'cv'].
    :param output_csv: Filepath of output csv.

    :return:  None.
    """
    makedirs([os.path.dirname(output_csv)])

    print(f'Compiling growing season effective precipitation vs water year precipitation into a csv...')

    # empty dictionary with to store data
    extract_dict = {'peff': [], 'precip': [],
                    'index': []}  # the index column keep track of pixel which will be used for aggregation to mean

    # basin name dict
    basin_name_dict = {'gmd4': 'GMD4, KS', 'gmd3': 'GMD3, KS', 'rpb': 'Republican Basin, CO',
                       'hqr': 'Harquahala INA, AZ', 'doug': 'Douglas AMA, AZ',
                       'dv': 'Diamond Valley, NV', 'cv': 'Central Valley, CA'}

    # lopping through each year and storing data in a list
    for year in years:
        peff_data = glob(os.path.join(basin_peff_dir, f'*{year}*.tif'))[0]
        precip_data = glob(os.path.join(basin_water_yr_precip_dir, f'*{year}*.tif'))[0]

        peff_arr = read_raster_arr_object(peff_data, get_file=False).flatten()
        precip_arr = read_raster_arr_object(precip_data, get_file=False).flatten()

        extract_dict['peff'].extend(list(peff_arr))
        extract_dict['precip'].extend(list(precip_arr))
        extract_dict['index'].extend(
            range(len(list(precip_arr))))  # for each data, the indexing will start from 0 up to the max num of pixels

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # aggregation by mean
    df_avg = df.groupby(by='index')[['peff', 'precip']].mean()  # groupby() mean avoid nan values (-9999) by default

    # assigning basin name
    df_avg['basin_code'] = [basin_code] * len(df_avg)
    df_avg['basin_name'] = [basin_name_dict[basin_code]] * len(df_avg)

    df_avg.dropna(inplace=True)
    df_avg.to_csv(output_csv, index=False)


def run_growS_peff_water_yr_precip_compilation(years, basin_shp, Peff_input_dir, basin_Peff_output_dir,
                                               precip_input_dir, basin_precip_output_dir,
                                               basin_code, output_csv, skip_processing=False):
    """
    Driver function for processing growing season effective precipitation and water year precipitation for each basin
    and compiling data into a csv.
    """
    if not skip_processing:
        clip_Peff_for_basin(years=years, basin_shp=basin_shp, Peff_input_dir=Peff_input_dir,
                            basin_Peff_output_dir=basin_Peff_output_dir, basin_code=basin_code,
                            month_range=None, resolution=model_res)
        clip_precip_for_basin(years=years, basin_shp=basin_shp, precip_input_dir=precip_input_dir,
                              basin_precip_output_dir=basin_precip_output_dir, basin_code=basin_code,
                              month_range=None, resolution=model_res)
        compile_basin_growS_peff_water_yr_precip_to_csv(years=years, basin_peff_dir=basin_Peff_output_dir,
                                                        basin_water_yr_precip_dir=basin_precip_output_dir,
                                                        basin_code=basin_code, output_csv=output_csv)
    else:
        pass


def compile_basin_monthly_peff_to_csv(years, month_range, basin_peff_dir, basin_code, output_csv):
    """
    Compiling pixel-wise monthly effective precipitation data for a basin into a csv.

    :param years: List of years_list to process data.
    :param month_range: Range of month to process data for. Example - (4, 10).
    :param basin_peff_dir: Basin monthly effective precipitation directory.
    :param basin_code: Basin keyword to add before the processed raster's name. Preferred to use from
                  ['gmd3', 'gmd4', 'rpb', 'hqr', 'doug', 'dv', 'cv'].
    :param output_csv: Filepath of output csv.

    :return:  None.
    """
    makedirs([os.path.dirname(output_csv)])

    print(f'Compiling monthly effective precipitation into a csv...')

    # empty dictionary with to store data
    extract_dict = {'peff': [], 'month': [],
                    'index': []}  # the index column keep track of pixel which will be used for aggregation to mean

    # basin name dict
    basin_name_dict = {'gmd4': 'GMD4, KS', 'gmd3': 'GMD3, KS', 'rpb': 'Republican Basin, CO',
                       'hqr': 'Harquahala INA, AZ', 'doug': 'Douglas AMA, AZ',
                       'dv': 'Diamond Valley, NV', 'cv': 'Central Valley, CA'}

    # lopping through each year and storing data in a list
    for year in years:
        months = list(range(month_range[0], month_range[1]+1))

        for month in months:
            peff_data = glob(os.path.join(basin_peff_dir, f'*{year}_{month}*.tif'))[0]

            peff_arr = read_raster_arr_object(peff_data, get_file=False).flatten()

            extract_dict['peff'].extend(list(peff_arr))
            extract_dict['index'].extend(range(len(list(peff_arr))))  # for each data, the indexing will start from 0 up to the max num of pixels

            month_list = [month] * len(list(peff_arr))
            extract_dict['month'].extend(month_list)

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # aggregation by mean
    df_avg = df.groupby(by=['index', 'month'])['peff'].mean()  # groupby() mean avoid nan values (-9999) by default

    # assigning month index as a month column
    df_avg = df_avg.reset_index()  # assigns index as columns
    df_avg = df_avg.drop(columns=['index'])

    # assigning basin name
    df_avg['basin_code'] = [basin_code] * len(df_avg)
    df_avg['basin_name'] = [basin_name_dict[basin_code]] * len(df_avg)

    df_avg.dropna(inplace=True)
    df_avg.to_csv(output_csv, index=False)


def compile_basin_monthly_precip_mv_avg_to_csv(years, month_range, basin_precip_dir, basin_code, output_csv):
    """
    Compiling pixel-wise monthly effective precipitation data for a basin into a csv.

    :param years: List of years_list to process data.
    :param month_range: Range of month to process data for. Example - (4, 10).
    :param basin_precip_dir: Basin monthly precipitation directory.
    :param basin_code: Basin keyword to add before the processed raster's name. Preferred to use from
                       ['gmd3', 'gmd4', 'rpb', 'hqr', 'doug', 'dv', 'cv'].
    :param output_csv: Filepath of output csv.

    :return:  None.
    """
    makedirs([os.path.dirname(output_csv)])

    print(f'Compiling monthly precipitation into a csv...')

    # empty dictionary with to store data
    extract_dict = {'precip': [], 'month': [],
                    'index': []}  # the index column keep track of pixel which will be used for aggregation to mean

    # basin name dict
    basin_name_dict = {'gmd4': 'GMD4, KS', 'gmd3': 'GMD3, KS', 'rpb': 'Republican Basin, CO',
                       'hqr': 'Harquahala INA, AZ', 'doug': 'Douglas AMA, AZ',
                       'dv': 'Diamond Valley, NV', 'cv': 'Central Valley, CA'}

    # lopping through each year and storing data in a list
    for year in years:
        months = list(range(month_range[0], month_range[1] + 1))

        for month in months:
            precip_data = glob(os.path.join(basin_precip_dir, f'*{year}_{month}*.tif'))[0]

            precip_arr = read_raster_arr_object(precip_data, get_file=False).flatten()

            extract_dict['precip'].extend(list(precip_arr))
            extract_dict['index'].extend(
                range(len(list(precip_arr))))  # for each data, the indexing will start from 0 up to the max num of pixels

            month_list = [month] * len(list(precip_arr))
            extract_dict['month'].extend(month_list)

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # assigning basin name
    df['basin_code'] = [basin_code] * len(df)
    df['basin_name'] = [basin_name_dict[basin_code]] * len(df)

    df.to_csv(output_csv, index=False)


def calc_moving_avg_monthly_precip(monthly_precip_csv, basin_code, output_csv):
    """
    Calculate moving average monthly precipitation for basins.
    Estimates the moving average for three default windows of 2, 3, and 4 months.

    :param monthly_precip_csv: Basin's monthly precipitation csv. Must have a index columns (representing consistent
                               pixel number
    :param basin_code: Basin keyword to add before the processed raster's name. Preferred to use from
                      ['gmd3', 'gmd4', 'rpb', 'hqr', 'doug', 'dv', 'cv'].
    :param output_csv: Filepath for output csv.

    :return: None.
    """
    # basin name dict
    basin_name_dict = {'gmd4': 'GMD4, KS', 'gmd3': 'GMD3, KS', 'rpb': 'Republican Basin, CO',
                       'hqr': 'Harquahala INA, AZ', 'doug': 'Douglas AMA, AZ',
                       'dv': 'Diamond Valley, NV', 'cv': 'Central Valley, CA'}

    # loading data
    df = pd.read_csv(monthly_precip_csv)
    df = df.dropna()

    # Sort the dataframe by index and month
    df_test = df.sort_values(by=['index', 'month'])

    # summing precip for all years_list for each month in a pixel.
    # Also, keeping count of total values summed for a pixel (each count value account for a year)
    df_test = df_test.groupby(['index', 'month'])['precip'].agg(['sum', 'count']).reset_index()

    # growing season months
    growS_months = list(range(4, 11))

    # empty dict to store the result
    results_dict = {'month': [], 'mean_precip_4_mnth': [], 'mean_precip_3_mnth': [], 'mean_precip_2_mnth': [],
                    'mean_precip': []}

    # we are only considering 2, 3, and 4 months including the current month as moving average window
    windows = [4, 3, 2, 1]

    # looping for each window, calculating moving average, and aggregating into a dataframe
    for window in windows:
        for mn in growS_months:
            if window == 4:  # moving season average for 4 months including the current month
                months_to_consider = list(range(mn - 3, mn + 1))
            elif window == 3:  # moving season average for 3 months including the current month
                months_to_consider = list(range(mn - 2, mn + 1))
            elif window == 2:  # moving season average for 2 months including the current month
                months_to_consider = list(range(mn - 1, mn + 1))
            else:
                months_to_consider = [mn]

            # filtering for months that fall inside the window
            df_filtered = df_test[df_test['month'].isin(months_to_consider)]

            # summing both the precipitation summed and the pixel count
            df_grouped = df_filtered.groupby('index').agg({'sum': 'sum', 'count': 'sum'}).reset_index()

            # mean precip for the given month window
            if window > 1:
                results_dict[f'mean_precip_{window}_mnth'].extend((df_grouped['sum'] / df_grouped['count']).values)
            else:
                results_dict['mean_precip'].extend((df_grouped['sum'] / df_grouped['count']).values)

            if window == 4:  # only recording month info once
                results_dict['month'].extend([mn] * len(df_grouped))

    df_done = pd.DataFrame(results_dict)

    # assigning basin code and name
    df_done['basin_code'] = [basin_code] * len(df_done)
    df_done['basin_name'] = [basin_name_dict[basin_code]] * len(df_done)

    df_done.to_csv(output_csv, index=False)


def run_monthly_peff_precip_compilation(years, basin_shp, Peff_input_dir, precip_input_dir,
                                        basin_Peff_output_dir, basin_precip_output_dir, basin_code,
                                        peff_output_csv, precip_output_csv, precip_moving_avg_output_csv,
                                        skip_processing=False):
    """
    Driver function for processing monthly effective precipitation and precipitation for each basin
    and compiling data into a csv.
    """
    if not skip_processing:
        # monthly effective precipitation
        clip_Peff_for_basin(years=years, basin_shp=basin_shp, Peff_input_dir=Peff_input_dir,
                            basin_Peff_output_dir=basin_Peff_output_dir, basin_code=basin_code,
                            month_range=(1, 12), resolution=model_res)
        compile_basin_monthly_peff_to_csv(years=years, month_range=(1, 12),
                                          basin_peff_dir=basin_Peff_output_dir,
                                          basin_code=basin_code, output_csv=peff_output_csv)

        # monthly precipitation (months 1-10 chosen as it will be used to estimate moving average of precip
        clip_precip_for_basin(years=years, basin_shp=basin_shp, basin_code=basin_code,
                              precip_input_dir=precip_input_dir, basin_precip_output_dir=basin_precip_output_dir,
                              month_range=(1, 12), resolution=model_res)

        compile_basin_monthly_precip_mv_avg_to_csv(years=years, month_range=(1, 12),
                                                   basin_precip_dir=basin_precip_output_dir,
                                                   basin_code=basin_code, output_csv=precip_output_csv)

        calc_moving_avg_monthly_precip(monthly_precip_csv=precip_output_csv, basin_code=basin_code,
                                       output_csv=precip_moving_avg_output_csv)
    else:
        pass


def compile_peff_precip_csv_all_basins(annual_csv_list, output_csv):
    """
        Compile csv files of all basins' effective precip into a single csv.

        :param annual_csv_list: List of basins' annual csv filepaths.
        :param output_csv: Filepath of compiled annual output csv.

        :return: None.
        """
    # making the output directory if not available
    makedirs([os.path.dirname(output_csv)])

    # empty dataframe to store the results
    compiled_annual_df = pd.DataFrame()

    for csv in annual_csv_list:
        df = pd.read_csv(csv)
        compiled_annual_df = pd.concat([compiled_annual_df, df])

    compiled_annual_df.to_csv(output_csv, index=False)