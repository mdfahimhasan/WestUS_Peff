import os
import numpy as np
from glob import glob
import rasterio as rio
from osgeo import gdal
import geopandas as gpd
from rasterio.merge import merge
from system_process import makedirs

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../Data/shapefiles/Western_US/Western_US_refraster_2km.tif'
conus_raster = '../Data/shapefiles/conus/conus_refraster_2km.tif'


def read_raster_arr_object(raster_file, rasterio_obj=False, band=1, get_file=True, change_dtype=True):
    """
    Get raster array and raster file.

    :param raster_file: Input raster file path.
    :param rasterio_obj: Set True if raster_file is a rasterio object.
    :param band: Selected band to read. Default set to 1.
    :param get_file: Set to False if raster file is not required.
    :param change_dtype: Set to True if want to change raster data type to float. Default set to True.

    :return: Raster numpy array and rasterio object file (get_file=True, rasterio_obj=False).
    """
    if not rasterio_obj:
        raster_file = rio.open(raster_file)
    else:
        get_file = False
    raster_arr = raster_file.read(band)
    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan
    if get_file:
        return raster_arr, raster_file
    else:
        return raster_arr


def write_array_to_raster(raster_arr, raster_file, transform, output_path, ref_file=None, nodata=no_data_value):
    """
    Write raster array to Geotiff format.

    :param raster_arr: Raster array data to be written.
    :param raster_file: Original rasterio raster file containing geo-coordinates.
    :param transform: Affine transformation matrix.
    :param output_path: Output file path.
    :param ref_file: Write output raster considering parameters from reference raster file.
    :param nodata: no_data_value set as -9999.

    :return: Output file path.
    """
    if ref_file:
        raster_file = rio.open(ref_file)
        transform = raster_file.transform

    with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=raster_arr.dtype,
            count=raster_file.count,
            crs=raster_file.crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(raster_arr, raster_file.count)

    return output_path


def mosaic_rasters(input_dir, output_dir, raster_name, ref_raster=WestUS_raster, search_by="*.tif",
                   resolution=model_res, nodata=no_data_value):
    """
    Mosaics multiple rasters into a single raster (rasters have to be in the same directory).

    :param input_dir: Input rasters' directory.
    :param output_dir: Output raster directory.
    :param raster_name: Output raster name.
    :param ref_raster: Reference raster file path. Set default to conus_raster.
    :param search_by: Input raster search criteria. Default set to '*tif'.
    :param resolution: Resolution of the output raster. Default set to 0.02000000000000000389 deg (2 km).
    :param nodata: no_data_value set as -9999.

    :return: Mosaiced raster array and file path of mosaiced raster.
    """
    input_rasters = glob(os.path.join(input_dir, search_by))
    raster_list = []
    for raster in input_rasters:
        arr, file = read_raster_arr_object(raster)
        raster_list.append(file)

    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    merged_arr, out_transform = merge(raster_list, bounds=ref_file.bounds, res=(resolution, resolution), nodata=nodata)

    merged_arr = np.where(ref_arr == 0, merged_arr, ref_arr)
    merged_arr = merged_arr.squeeze()

    makedirs([output_dir])
    out_raster = os.path.join(output_dir, raster_name)
    write_array_to_raster(raster_arr=merged_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=out_raster, nodata=nodata, ref_file=ref_raster)

    return merged_arr, out_raster


def clip_resample_reproject_raster(input_raster, input_shape, keyword, output_raster_dir, clip=False, resample=False,
                                   clip_and_resample=True, targetaligned=True, resample_algorithm='near',
                                   resolution=None, crs='EPSG:4269', output_datatype=gdal.GDT_Float32):
    """
    Clips, resamples, reprojects a given raster using input shapefile, resolution, and crs.

    ** If resolution is None, must provide a ref_raster. One of resolution/ref_raster must be available.

    :param input_raster: Input raster filepath.
    :param input_shape: Input shape filepath. Set to None when resample=True.
    :param keyword: 'Str' keyword to attach in front of processed raster. Can be ' '.
    :param output_raster_dir: Outpur directory filepath.
    :param clip: Set to True to clip only. When True, resample and clip_and_resample should be False. resolution can be
                 used as None (Default) in this case.
    :param resample: Set to True to resample only. When True, clip and clip_and_resample should be False. Need to set
                     a resolution.
    :param clip_and_resample: Set to True to both clip and resample. When True, clip and resample should be False. Need
                              to set a resolution. Set input_shape to None.
    :param targetaligned: Set to False if pixels don't need to be aligned in case of clip_and_resample=True. Look into
                          the result when using this to be sure about expected outcome.
    :param resample_algorithm: Resample algorithm to use in resampling. Can take near/bilinear/average/mode/max/min etc.
                               Default is 'near'.
    :param resolution: Output raster resolution. Default set to None (for clip=True). For other purposes, use specific
                       resolution value (for example, 0.02000000000000000389 deg).
    :param crs: Output raster projection. Default set to 'EPSG:4269' (NAD83).
    :param output_datatype: Output data type. Default set to gdal.GDT_Float32.

    :return: Processed raster filepath.
    """
    raster_name = os.path.basename(input_raster)
    output_raster_name = keyword + '_' + raster_name
    if keyword == ' ':
        output_raster_name = raster_name

    makedirs([output_raster_dir])
    output_filepath = os.path.join(output_raster_dir, output_raster_name)

    raster_file = gdal.Open(input_raster)

    if clip:  # set resample, clip_and_resample = False
        # resolution can be None in argument
        _, xres, _, _, _, yres = raster_file.GetGeoTransform()
        processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                   targetAlignedPixels=targetaligned, xRes=xres, yRes=yres, cutlineDSName=input_shape,
                                   cropToCutline=True, dstNodata=no_data_value, outputType=output_datatype)

    elif resample:  # set clip, clip_and_resample = False
        # have to provide a resolution value in argument
        # input_shape can be set to None
        processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                   targetAlignedPixels=targetaligned, xRes=resolution, yRes=resolution,
                                   dstNodata=no_data_value, resampleAlg=resample_algorithm, outputType=output_datatype)

    elif clip_and_resample:  # set clip, resample = False
        # argument must have input_shape and resolution value
        processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                   targetAlignedPixels=targetaligned, xRes=resolution, yRes=resolution,
                                   cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                   resampleAlg=resample_algorithm, outputType=output_datatype)

    del processed_data

    return output_filepath

