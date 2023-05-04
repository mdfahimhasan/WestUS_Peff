import os
import numpy as np
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgpd
from dask import dataframe as ddf
from osgeo import gdal, osr, ogr

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def create_buffer(input_shapefile, distance, output_shapefile, change_crs='EPSG:32611'):
    """
    creates a buffer region around a shapefile.

    :param input_shapefile: Filepath of input shapefile.
    :param distance: Value of distance. Unit will be in the unit of the crs.
    :param output_shapefile: Filepath of output (buffered) shapefile.
    :param change_crs: Default set to 'EPSG:32611' (UTM zone 11N) as the projected crs.

    :return: Returns the filepath of buffered shapefile.
    """
    input_gdf = gpd.read_file(input_shapefile)
    original_crs = input_gdf.crs
    if change_crs is not None:
        input_gdf = input_gdf.to_crs(change_crs)

    input_gdf = input_gdf.buffer(distance)
    input_gdf = input_gdf.to_crs(original_crs)  # converting back to original crs
    input_gdf.to_file(output_shapefile)

    return output_shapefile


def clip_vector(input_shapefile, mask_shapefile, output_shapefile, create_zero_buffer=False):
    """
    Clips a vector file based on the vector mask provided.

    :param input_shapefile: Filepath of input shapefile.
    :param mask_shapefile: Filepath of shapefile to use as mask.
    :param output_shapefile: Filepath of output (clipped) shapefile.
    :param create_zero_buffer: Set to True to create zero buffer around input shapefile. Default set to False.
                               There might be intersection error for complex/problematic shapefiles. Creating a zero
                               buffer around it solves the issue. If it doesn't further investigate.

    :return: Returns the filepath of clipped shapefile.
    """
    if create_zero_buffer:
        # # Buffering might ruin the attribute information of the input shapefile. Do further processing.

        output_dir = os.path.dirname(output_shapefile)
        buffered_shape_path = os.path.join(output_dir, 'zero_buffer.shp')
        # Creating zero buffered shapefile. The crs will be converted to original crs of the shapefile automatically
        buffered_shapefile = create_buffer(input_shapefile, distance=0, output_shapefile=buffered_shape_path,
                                           change_crs='EPSG:32611')
        input_shapefile = buffered_shapefile

    input_gdf = gpd.read_file(input_shapefile)
    mask_gdf = gpd.read_file(mask_shapefile)

    clipped_gdf = gpd.clip(input_gdf, mask_gdf, keep_geom_type=True)

    clipped_gdf.to_file(output_shapefile)

    return output_shapefile


def add_attr_to_county_fromCSV(input_shapefile, attr_csv_data, output_shapefile, year_filter, columns_to_keep):
    """
    Add attribute information to county shapefile.

    :param input_shapefile: Filepath of input shapefile.
    :param attr_csv_data: Filepath of attribute data for each county at csv format.
    :param output_shapefile: Filepath of output shapefile.
    :param year_filter: int. Year to filter csv data.
    :param columns_to_keep: Tuple of columns to keep in the final shapefile.

    :return: A new shapefile with added attribute information.
    """
    gw_df = pd.read_csv(attr_csv_data)
    gw_df = gw_df[gw_df['Year'] == year_filter]

    county_gdf = gpd.read_file(input_shapefile)
    county_gdf = county_gdf.merge(gw_df, on='fips', how='left')

    keep_columns = list(columns_to_keep)
    keep_columns.append('geometry')
    county_gdf = county_gdf[keep_columns]

    county_gdf.to_file(output_shapefile)

    return output_shapefile


def create_pixel_multipoly_shapefile(refraster, interim_output_raster, output_file):
    """
    Creates a shapefile of polygon each having a width of the input raster's pixel size.
    ** This code will generate the polygon for the whole extent of the input raster file. Clip it with appropriate
    shapefile in python/gis to get the intended polygon shapefile.

    :param refraster: Filepath of reference raster.
    :param interim_output_raster: Filepath of intermediate output raster where each pixel has unique DNvalue.
    :param output_file: Filepath of output polygon.

    :return: A multi-polygon with each polygon having a width of the input raster's pixel size.
    """

    # getting total polygons estimate
    ref_arr, ref_file = read_raster_arr_object(refraster)
    shape = ref_arr.shape
    total_pol = (shape[0] * shape[1]) + 1  # number of total polygons to create based on no. members in ref raster

    # the new_arr will have individual pixels with unique DN values
    new_arr = np.arange(start=1, stop=total_pol, step=1).reshape(shape)

    # creating scratch dir in case the folder doesn't exist
    makedirs(['../../scratch'])
    write_array_to_raster(new_arr, ref_file, ref_file.transform, interim_output_raster)

    # converting the newly created raster to polygon
    raster = gdal.Open(interim_output_raster)
    band = raster.GetRasterBand(1)
    arr = band.ReadAsArray()

    proj = raster.GetProjection()
    shp_proj = osr.SpatialReference()
    shp_proj.ImportFromWkt(proj)

    call_drive = ogr.GetDriverByName('ESRI Shapefile')
    create_shp = call_drive.CreateDataSource(output_file)
    shp_layer = create_shp.CreateLayer('layername', srs=shp_proj)
    new_field = ogr.FieldDefn(str('ID'), ogr.OFTInteger)
    shp_layer.CreateField(new_field)

    gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)
    create_shp.Destroy()

    raster = None


def extract_centroid_of_polygon(input_shapefile, output_point_shpafile, id_col='ID'):
    input_gdf = dgpd.read_file(input_shapefile, npartitions=4)

    centroid = input_gdf.geometry.centroid.compute().to_crs(input_gdf.crs)
    id = input_gdf[id_col]

    data = {'ID': id, 'geometry': centroid}

    point_gdf = gpd.GeoDataFrame(data, geometry='geometry')
    point_gdf.to_file(output_point_shpafile)
