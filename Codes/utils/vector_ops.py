import os
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
import dask_geopandas as dgpd
from osgeo import gdal, osr, ogr
from shapely.geometry import Polygon

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


def add_attr_to_county_fromCSV(input_shapefile, attr_csv_df_data, output_shapefile, year_filter,
                               columns_to_keep=None):
    """
    Add attribute information to county shapefile.

    :param input_shapefile: Filepath of input shapefile.
    :param attr_csv_df_data: Filepath of attribute data for each county at csv/dataframe format.
    :param output_shapefile: Filepath of output shapefile.
    :param year_filter: int. Year to filter csv data.
    :param columns_to_keep: Tuple of columns to keep in the final shapefile. Default set to None.

    :return: A new shapefile with added attribute information.
    """
    if '.csv' in attr_csv_df_data:
        df = pd.read_csv(attr_csv_df_data)
    else:
        df = attr_csv_df_data

    df = df[df['Year'] == year_filter]

    county_gdf = gpd.read_file(input_shapefile)
    county_gdf = county_gdf.merge(df, on='fips', how='left')

    if columns_to_keep is not None:  # this block filters in certain columns if columns_to_keep is given
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
    :param interim_output_raster: Filepath of intermediate output raster where each pixel has unique DNv alue.
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

###########
# # # This function isn't optimized and need major change/modification.

def raster_to_shapefile(input_raster, output_shapefile, shapefile_crs=None):
    """
    Convert a raster to shapefile.
    * This code is fitted to convert raster with binary data, e.g., 0 and 1.

    :param input_raster: Filepath of input raster.
    :param output_shapefile: Filepath of output polygon shapefile.
    :param shapefile_crs: The output shapefile crs in EPSG format, e.g., 'EPSG:4269'.
                          Default set to None to set projection from the input_raster.

    :return: A polygon shapefile.
    """
    # opening raster
    raster = gdal.Open(input_raster)
    band = raster.GetRasterBand(1)

    # Projection
    if shapefile_crs is None:
        proj = raster.GetProjection()
        shp_proj = osr.SpatialReference()
        shp_proj.ImportFromWkt(proj)
    else:
        proj = CRS.from_string(shapefile_crs)
        shp_proj = osr.SpatialReference()
        shp_proj.ImportFromWkt(proj)

    # Creating Polygon shapefile holder
    call_drive = ogr.GetDriverByName('ESRI Shapefile')
    create_shp = call_drive.CreateDataSource(output_shapefile)
    shp_layer = create_shp.CreateLayer('layername', srs=shp_proj)
    new_field = ogr.FieldDefn(str('ID'), ogr.OFTInteger)
    shp_layer.CreateField(new_field)

    # Polygonize
    gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)
    create_shp.Destroy()

    raster = None
###########


def extract_centroid_of_polygon(input_shapefile, output_point_shpafile, id_col='ID'):
    input_gdf = dgpd.read_file(input_shapefile, npartitions=4)

    centroid = input_gdf.geometry.centroid.compute().to_crs(input_gdf.crs)
    id = input_gdf[id_col]

    data = {'ID': id, 'geometry': centroid}

    point_gdf = gpd.GeoDataFrame(data, geometry='geometry')
    point_gdf.to_file(output_point_shpafile)


def create_fishnets_from_shapefile(input_shape, num_cols, num_rows, output_shape, crs=None):
    """
    Create fishnet polygons' shapefile using the extent of input shapefile.

    :param input_shape: Filepath of input shapefile from which extent will be extracted.
    :param num_cols: Number of columns in the fishnet.
    :param num_rows: Number of rowss in the fishnet.
    :param output_shape: Filepath of output fishnet shapefile.
    :param crs: CRS in 'EPSG:4269' format. Default set to None to use crs from input shapefile.

    :return: None.
    """
    input_shp_gdf = gpd.read_file(input_shape)
    xmin, ymin, xmax, ymax = input_shp_gdf.total_bounds

    grid_x_size = abs((xmin - xmax) / num_cols)
    grid_y_size = abs((ymin - ymax) / num_rows)

    lats = list(np.arange(xmin, (xmax + grid_x_size), grid_x_size))
    lons = list(np.arange(ymin, (ymax + grid_y_size), grid_y_size))

    poly_geoms = []
    for i in range(len(lats)-1):
        for j in range(len(lons)-1):
            bounds = Polygon([(lats[i], lons[j]), (lats[i], lons[j+1]), (lats[i+1], lons[j+1]), (lats[i+1], lons[j]),
                             (lats[i], lons[j])])
            poly_geoms.append(bounds)

    if crs is None:
        crs = input_shp_gdf.crs

    fishnet = gpd.GeoDataFrame(poly_geoms, columns=['geometry']).set_crs(crs)
    fishnet.to_file(output_shape)


# create_fishnets_from_shapefile(input_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp',
#                                num_cols=50, num_rows=25,
#                                output_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m.shp')


