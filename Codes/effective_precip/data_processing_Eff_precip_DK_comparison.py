import os
import sys
import geopandas as gpd
from glob import glob

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object

# Randomized point shapefiles
random_point_shapefile = glob(os.path.join('../../Data_main/shapefiles/Effective_precip_comparison/Random_points_shape', '*.shp'))

# The shapefiles from GEE has been downloaded with DK's effective precip and openET 30m values for month 6, 2016
# So, we will only extract area averaged rainfed cropET information for month 6, 2016
rainfed_cropET_monthly_dir = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly'
rainfed_cropET_2016_6 = glob(os.path.join(rainfed_cropET_monthly_dir, '*2016_6*.tif'))[0]

rainfed_cropET_arr, rainfed_cropET_file = read_raster_arr_object(rainfed_cropET_2016_6)

shape_outdir = '../../Data_main/shapefiles/Effective_precip_comparison/Random_points_shape_updated'
makedirs([shape_outdir])

# ##############################################################
# # # Process switches for adding data to the shapefiles
skip_add_data_to_shapefile = True
# ##############################################################

if not skip_add_data_to_shapefile:
    for shape in random_point_shapefile:
        shape_name = os.path.basename(shape)

        shape_gdf = gpd.read_file(shape)
        shape_gdf = shape_gdf.rename(columns={'b1': 'Ef_Pr_DK'})

        shape_gdf['lat'] = shape_gdf['geometry'].y
        shape_gdf['lon'] = shape_gdf['geometry'].x

        # extracting pixel info from rainfed cropET and storing in a list
        rainfed_cropET_val = []
        for index, row in shape_gdf.iterrows():
            x, y = rainfed_cropET_file.index(row['lon'], row['lat'])
            et_value = rainfed_cropET_arr[x, y]
            rainfed_cropET_val.append(et_value)

        shape_gdf['Rain_ET'] = rainfed_cropET_val
        shape_gdf = shape_gdf.dropna()

        shape_gdf.to_file(os.path.join(shape_outdir, shape_name))