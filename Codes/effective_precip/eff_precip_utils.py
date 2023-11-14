import os
import sys
import numpy as np
from glob import glob
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Polygon

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def filter_effective_precip_training_data(training_zone_shp, general_output_dir, refraster=WestUS_raster,
                                          resolution=model_res, skip_processing=False):
    """
    Filter the rainfed cropET training data by defined filters and bounding boxes.

    :param training_zone_shp: Input shapefile path of the training bounding box zones.
    :param general_output_dir: Filepath of general output directory. The final output dir will be configured inside
                                the function.
    :param refraster: Default set to Western US reference raster.
    :param resolution: Default set to model resolution.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        # years and months to process over
        years = [2016, 2017, 2018, 2019, 2020]
        months = list(range(1, 13))

        # training zone bounding box gdf
        training_zone_gdf = gpd.read_file(training_zone_shp)

        # reference raster
        ref_arr, ref_file = read_raster_arr_object(refraster)
        total_bounds = ref_file.bounds

        # primary and secondary output directory creation
        bbox_temp_output_dir = os.path.join(general_output_dir, 'bbox_temp_raster')
        cropET_interim_bbox_dir = os.path.join(general_output_dir, 'cropET_interim_for_bbox')
        final_filtered_cropET_dir = os.path.join(general_output_dir, 'final_filtered_cropET_for_training')

        makedirs([general_output_dir, bbox_temp_output_dir, cropET_interim_bbox_dir, final_filtered_cropET_dir])

        for id, type, geom in zip(training_zone_gdf['id'], training_zone_gdf['train_type'], training_zone_gdf['geometry']):
            # creating shapefile for each of the training boxes and rasterizing the shapefile such that values inside the
            # box is 1 and outside the box is no data for that respective box

            # creating polygon shapefile for each training box
            minx, miny, maxx, maxy = geom.bounds
            bbox = Polygon(((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)))
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:4269')

            bbox_shapefile = os.path.join(bbox_temp_output_dir, f'bbox_{id}.shp')
            bbox_gdf.to_file(bbox_shapefile)

            # rasterizing each training box
            bbox_raster = os.path.join(bbox_temp_output_dir, f'bbox_raster_{id}_type{type}.tif')
            raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                                   outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                                   noData=no_data_value, burnValues=1, attribute=None, allTouched=True)
            gdal.Rasterize(destNameOrDestDS=bbox_raster, srcDS=bbox_shapefile, options=raster_options,
                           resolution=resolution)

        rainfed_cropET_dir = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly'
        irrigated_cropET_dir = '../../Data_main/Raster_data/Irrigated_cropET/WestUS_monthly'
        rainfed_frac_dir = '../../Data_main/Raster_data/Rainfed_cropland/Rainfed_Frac'
        irrigated_frac_dir = '../../Data_main/Raster_data/Irrigated_cropland/Irrigated_Frac'
        rainfed_cropland_dir = '../../Data_main/Raster_data/Rainfed_cropland'
        irrigated_cropland_dir = '../../Data_main/Raster_data/Irrigated_cropland'
        usda_cdl_dir = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'
        excess_ET_filter_dir = '../../Data_main/Raster_data/Excess_ET_filter'

        # collecting all rainfed cropET dataset
        rainfed_cropET_datasets = glob(os.path.join(rainfed_cropET_dir, '*.tif'))

        # going over each rainfed cropET data and applying the rasterized bbox raster on it
        # this will create individual raster for each cropET raster for each bounding box
        for cropET in rainfed_cropET_datasets:
            cropET_arr, file = read_raster_arr_object(cropET)

            year = os.path.basename(cropET).split('_')[2]
            month = os.path.basename(cropET).split('_')[3].split('.')[0]

            irrigated_cropET_data = glob(os.path.join(irrigated_cropET_dir, f'*{year}_{month}*.tif'))[0]
            rainfed_frac_data = glob(os.path.join(rainfed_frac_dir, f'*{year}*.tif'))[0]
            irrigated_frac_data = glob(os.path.join(irrigated_frac_dir, f'*{year}*.tif'))[0]
            rainfed_cropland_data = glob(os.path.join(rainfed_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            cdl_data = glob(os.path.join(usda_cdl_dir, f'*{year}*.tif'))[0]
            excess_et_filter_data = glob(os.path.join(excess_ET_filter_dir, f'*{year}*.tif'))[0]

            irrigated_cropET_arr = read_raster_arr_object(irrigated_cropET_data, get_file=False)
            rainfed_frac_arr = read_raster_arr_object(rainfed_frac_data, get_file=False)
            irrigated_frac_arr = read_raster_arr_object(irrigated_frac_data, get_file=False)
            rainfed_cropland_arr =  read_raster_arr_object(rainfed_cropland_data, get_file=False)
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)
            cdl_arr = read_raster_arr_object(cdl_data, get_file=False)
            excess_et_arr = read_raster_arr_object(excess_et_filter_data, get_file=False)

            # training zone types -
            # type 1: pasture lands with rainfed croplands or any rainfed cropland with no overlapping with irrigated croplands,
            #         excess_et_filter = 1 in both cases
            # type 2: irrigated pixels where rainfed fraction < irrigated fraction, rainfed fraction > 20%,
            #         rainfed fraction < 60% of irrigation fraction, rainfed fraction < irrigation fraction,
            #         rainfed cropET < irrigated cropET, excess_et_filter = 1,

            # collecting the bbox rasters and looping over them to apply each bbox raster on a cropET raster
            bbox_rasters = glob(os.path.join(bbox_temp_output_dir, '*.tif'))

            for bbox_ras in bbox_rasters:
                filtered_cropET_arr = None

                box_id = os.path.basename(bbox_ras).split('_')[2]
                bbox_arr = read_raster_arr_object(bbox_ras, get_file=False)

                if 'type1' in bbox_ras:
                    filtered_cropET_arr = np.where(((cdl_arr == 176) & (rainfed_cropland_arr == 1) & (excess_et_arr == 1)) |
                                                   ((rainfed_cropland_arr == 1) & (irrigated_cropland_arr == -9999) &
                                                    (excess_et_arr == 1)), cropET_arr, -9999)

                elif 'type2' in bbox_ras:
                    filtered_cropET_arr = np.where((irrigated_cropland_arr == 1) & (rainfed_cropland_arr == 1) &
                                                    (rainfed_frac_arr > (0.60 * irrigated_frac_arr)) &
                                                     (rainfed_frac_arr < irrigated_frac_arr) &
                                                     (irrigated_cropET_arr > cropET_arr) & (excess_et_arr == 1),
                                                     cropET_arr, -9999)

                # applying the bbox raster on the filtered cropET so that everything outside the box becomes nodata
                final_cropET_arr = filtered_cropET_arr * bbox_arr

                # saving the cropET raster for each year-month for each bbox
                output_cropET = os.path.join(cropET_interim_bbox_dir, f'cropET_bbox_{box_id}_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_cropET_arr, raster_file=file, transform=file.transform,
                                      output_path=output_cropET)

        # compiling final filtered training data
        for year in years:
            for month in months:
                print(f'compiling filtered effect. precip training data for year {year}, month {month}...')

                # collecting data for all boxes for a single year-month
                # the  dot (.) is important to collect data for the same months. Not providing 'tif' here as glob will find
                # the rasters without it and we can't use both 'dot' and 'tif' for our purpose at the same time.
                filtered_training_rasters = glob(os.path.join(cropET_interim_bbox_dir, f'*{year}_{month}.*'))

                final_train_arr = None  # we will iterate and paste values of each training box here

                # compiling the individual bbox cropET raster in a single raster for each year-month
                for each in filtered_training_rasters:
                    if each == filtered_training_rasters[0]:
                        arr = read_raster_arr_object(each, get_file=False)
                        final_train_arr = arr

                    else:
                        arr = read_raster_arr_object(each, get_file=False)
                        final_train_arr = np.where(~np.isnan(final_train_arr), final_train_arr, arr)

                output_raster_name = os.path.join(final_filtered_cropET_dir, f'CropET_for_training_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_train_arr, raster_file=ref_file, transform=ref_file.transform,
                                      output_path=output_raster_name)
    else:
        pass


# # Filtering training data
training_zone_shapefile = '../../Data_main/shapefiles/Western_US_ref_shapes/effective_precip_training_zones.shp'
output_dir = '../../Data_main/Raster_data/Rainfed_cropET_filtered_training'
skip_effective_precip_training_data_filtering = True

filter_effective_precip_training_data(training_zone_shp=training_zone_shapefile,
                                      general_output_dir=output_dir,
                                      refraster=WestUS_raster,
                                      resolution=model_res,
                                      skip_processing=skip_effective_precip_training_data_filtering)