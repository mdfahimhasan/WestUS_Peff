
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 00:27:13 2021

@author: Dawit Wolday; modified Md Fahim Hasan
"""

import os
import ee
import requests
import geopandas as gpd


# # Landsat Cloudmask
def cloudmask_Landsat(image):
    """
    Removing cloudmask from landsat 05 and 08 data.

    param : {ee.Image} image input landsat SR image
    return : {ee.Image} cloudmasked landsat image
    """
    # Bits 0, 1 are cloud state and Bits 2 is cloud shadow.
    cloudMask0 = (1 << 3)
    cloudMask1 = (1 << 4)

    qa = image.select('QA_PIXEL')
    # Both flags should be set to zero, indicating clear conditions.
    Mask = qa.bitwiseAnd(cloudMask0).eq(0).And(qa.bitwiseAnd(cloudMask1).eq(0))
    return image.updateMask(Mask)


def download_Landsat05(aoi_shape_file, outdir,
                       year_list=[1990, 1998, 2006], start_month=1, end_month=12,
                       ):
    """
    :param year_list: List of years in %Y format
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param aoi_shp_file: Area of interest shapefile (must be in WGS84)
    :param outdir: Download directory
    :return: None
    """

    ee.Initialize()

    landsat= ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
    aoi_shp = gpd.read_file(aoi_shape_file)
    grid_geom = aoi_shp['geometry']
    grid_no = aoi_shp['grid_no']

    for grid, serial in zip(grid_geom, grid_no):
        roi = grid.bounds
        gee_aoi = ee.Geometry.Rectangle(roi)
        for year in year_list:
            start_date = ee.Date.fromYMD(year, start_month, 1)
            if end_month == 12:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)
            else:
                end_date = ee.Date.fromYMD(year, end_month + 1, 1)
            if end_month <= start_month:
                start_date = ee.Date.fromYMD(year - 1, start_month, 1)

            SR_B1Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B1').\
                median().multiply(0.0000275).add(-0.2).toDouble()
            SR_B2Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B2').\
                median().multiply(0.0000275).add(-0.2).toDouble()
            SR_B3Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B3').\
                median().multiply(0.0000275).add(-0.2).toDouble()
            SR_B4Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B4'). \
                median().multiply(0.0000275).add(-0.2).toDouble()

            SR_B1Total_url = SR_B1Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })

            SR_B2Total_url = SR_B2Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })

            SR_B3Total_url = SR_B3Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })
            SR_B4Total_url = SR_B4Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })

            gee_vars = ['SR_B1_', 'SR_B2_', 'SR_B3_', 'SR_B4_']
            gee_links = [SR_B1Total_url,SR_B2Total_url, SR_B3Total_url, SR_B4Total_url]
            for gee_var, gee_url in zip(gee_vars, gee_links):
                local_file_name = os.path.join(outdir, gee_var + str(year) + '_' + str(serial)+ '.tif')
                print('Dowloading', local_file_name, '...')
                r = requests.get(gee_url, allow_redirects=True)
                open(local_file_name, 'wb').write(r.content)


def download_Landsat08(aoi_shape_file, outdir,
                       year_list=[2014, 2022], start_month=1, end_month=12,
                       ):
    """

    :param year_list: List of years in %Y format
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param aoi_shp_file: Area of interest shapefile (must be in WGS84)
    :param outdir: Download directory
    :return: None
    """

    ee.Initialize()

    landsat= ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    aoi_shp = gpd.read_file(aoi_shape_file)
    grid_geom = aoi_shp['geometry']
    grid_no = aoi_shp['grid_no']

    for grid, serial in zip(grid_geom, grid_no):
        roi = grid.bounds
        gee_aoi = ee.Geometry.Rectangle(roi)
        for year in year_list:
            start_date = ee.Date.fromYMD(year, start_month, 1)
            if end_month == 12:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)
            else:
                end_date = ee.Date.fromYMD(year, end_month + 1, 1)
            if end_month <= start_month:
                start_date = ee.Date.fromYMD(year - 1, start_month, 1)

            SR_B2Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B2'). \
                median().multiply(0.0000275).add(-0.2).toDouble()
            SR_B3Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B3').\
                median().multiply(0.0000275).add(-0.2).toDouble()
            SR_B4Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B4').\
                median().multiply(0.0000275).add(-0.2).toDouble()
            SR_B5Total = landsat.filterDate(start_date, end_date).map(cloudmask_Landsat).select('SR_B5').\
                median().multiply(0.0000275).add(-0.2).toDouble()

            SR_B2Total_url = SR_B2Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })
            SR_B3Total_url = SR_B3Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })

            SR_B4Total_url = SR_B4Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })

            SR_B5Total_url = SR_B5Total.getDownloadUrl({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': gee_aoi,
                'format': 'GEO_TIFF'
            })

            gee_vars = ['SR_B2_', 'SR_B3_', 'SR_B4_', 'SR_B5_']
            gee_links = [SR_B2Total_url, SR_B3Total_url,SR_B4Total_url, SR_B5Total_url]
            for gee_var, gee_url in zip(gee_vars, gee_links):
                local_file_name = os.path.join(outdir, gee_var + str(year) + '_' + str(serial)+ '.tif')
                print('Dowloading', local_file_name, '...')
                r = requests.get(gee_url, allow_redirects=True)
                open(local_file_name, 'wb').write(r.content)


# download_Landsat05(aoi_shape_file='F:/WestUS_Wateruse_SpatialDist/Codes/Tomalika/sudarban_shp/Sundarban_eco_region_shp.shp',
#                    outdir=r'F:\WestUS_Wateruse_SpatialDist\Codes\Tomalika\sudarban_shp\downloaded')

# download_Landsat08(aoi_shape_file='F:/WestUS_Wateruse_SpatialDist/Codes/Tomalika/sudarban_shp/Sundarban_eco_region_shp.shp',
#                    outdir=r'F:\WestUS_Wateruse_SpatialDist\Codes\Tomalika\sudarban_shp\downloaded')


