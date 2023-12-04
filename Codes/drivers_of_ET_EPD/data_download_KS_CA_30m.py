import os
import ee
import sys
import requests
import geopandas as gpd

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.data_download_preprocess.download import get_gee_dict
from Codes.utils.raster_ops import read_raster_arr_object

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def download_openET_30m(data_name, years, months, grid_shape, download_dir):
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    month_list = [m for m in range(months[0], months[1] + 1)]  # creating list of months

    # getting imagecollection name, band names, reducer..
    data, band, multiply_scale, reducer, _, _, _, _ = get_gee_dict(data_name)

    for year in years:  # first loop for years
        for month in month_list:  # second loop for months
            print('********')
            print(f'Getting OpenET data for year={year}, month={month}.....')

            # Setting date ranges
            start_date = ee.Date.fromYMD(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)

            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year

            download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date)\
                                .reduce(reducer).toFloat()

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                data_url = download_data.getDownloadURL({'name': data_name,
                                                         'crs': 'EPSG:4269',  # NAD83
                                                         'scale': 30,
                                                         'region': gee_extent,
                                                         'format': 'GEO_TIFF'})

                r = requests.get(data_url, allow_redirects=True)

                output_raster = os.path.join(download_dir, f'{data_name}_{year}_{month}_{grid_sr}.tif')
                print('Downloading', output_raster, '.....')

                # save data to local file path
                open(output_raster, 'wb').write(r.content)

                # This is a check block to see if downloaded datasets are OK
                # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
                # Re-downloading the same data might not have that error
                try:
                    arr = read_raster_arr_object(output_raster, get_file=False)

                except:
                    print(f'Downloaded data corrupted. Re-downloading {output_raster}.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(output_raster, 'wb').write(r.content)


def download_irrigated_cropland_30m(data_name, years, grid_shape, download_dir):
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    if 'KS' in data_name:  # for GMD, kansas
        # Extracting dataset information required for downloading from GEE
        lanid_asset, _, _, _, _, _, _, _ = get_gee_dict('LANID')
        lanid_data_band_dict = {2000: 'lanid_2000', 2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003',
                                2004: 'lanid_2004', 2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007',
                                2008: 'lanid_2008', 2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011',
                                2012: 'lanid_2012', 2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015',
                                2016: 'lanid_2016', 2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019',
                                2020: 'lanid_2020'}

        # Extracting AIM-HPA dataset information (saved as an asset) from GEE
        AimHpa_asset, _, _, _, _, _, _, _ = get_gee_dict('AIM-HPA')
        aim_hpa_data_band_dict = {2000: 'aim_2000', 2001: 'aim_2001', 2002: 'aim_2002', 2003: 'aim_2003',
                                  2004: 'aim_2004', 2005: 'aim_2005', 2006: 'aim_2006', 2007: 'aim_2007',
                                  2008: 'aim_2008', 2009: 'aim_2009', 2010: 'aim_2010', 2011: 'aim_2011',
                                  2012: 'aim_2012', 2013: 'aim_2013', 2014: 'aim_2014', 2015: 'aim_2015',
                                  2016: 'aim_2016', 2017: 'aim_2017', 2018: 'aim_2018', 2019: 'aim_2019',
                                  2020: 'aim_2020'}

        for year in years:  # first loop for years
            print('********')
            print(f'Getting Irrigated Cropland data for year={year}.....')

            # ######### Merging irrigated field dataset from LANID and AIM-HPA
            # # LANID data for the year
            # In LANID dataset irrigated fields are assigned as 1
            lanid_band = lanid_data_band_dict[year]
            irr_lanid = ee.Image(lanid_asset).select(lanid_band)
            irr_lanid = irr_lanid.eq(1).unmask()  # unmask() assigns zero value to no data pixels

            # # AIM-HPA data for the year
            # In AIM-HPA dataset irrigated fields are assigned as 1
            hpa_band = aim_hpa_data_band_dict[year]
            irr_hpa = ee.Image(AimHpa_asset).select(hpa_band)
            irr_hpa = irr_hpa.eq(1).unmask()  # unmask() assigns zero value to no data pixels

            # # Joining LANID and AIM-HPA
            # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
            # Remapping the irrigated 0 values to 1 (others to 0) and then applying a mask
            irr_total = irr_lanid.add(irr_hpa)
            irr_total = irr_total.eq(0).remap([0, 1], [1, 0])
            irr_total = irr_total.updateMask(irr_total)

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                data_url = irr_total.getDownloadURL({'name': data_name,
                                                     'crs': 'EPSG:4269',  # NAD83
                                                     'scale': 30,  # in meter
                                                     'region': gee_extent,
                                                     'format': 'GEO_TIFF'})

                r = requests.get(data_url, allow_redirects=True)

                output_raster = os.path.join(download_dir, f'{data_name}_{year}_{grid_sr}.tif')
                print('Downloading', output_raster, '.....')

                # save data to local file path
                open(output_raster, 'wb').write(r.content)

                # This is a check block to see if downloaded datasets are OK
                # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
                # Re-downloading the same data might not have that error
                try:
                    arr = read_raster_arr_object(output_raster, get_file=False)

                except:
                    print(f'Downloaded data corrupted. Re-downloading {output_raster}.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(output_raster, 'wb').write(r.content)

    elif 'CA' in data_name:  # for Central Valley, CA
        # Extracting dataset information required for downloading from GEE
        irrmapper_asset, irrmapper_band, multiply_scale, reducer, _, _, _, _= get_gee_dict('IrrMapper')

        for year in years:  # first loop for years
            print('********')
            print(f'Getting Irrigated Cropland data for year={year}.....')

            # Filtering data for the year range and reducing data
            irrmap = ee.ImageCollection(irrmapper_asset).filter(ee.Filter.calendarRange(year, year, 'year'))\
                                        .select(irrmapper_band).max().toFloat()

            # In IrrMapper dataset irrigated fields are assigned as 0
            # Converting the irrigated values to 1 and setting others as 0
            mask = irrmap.eq(0)
            irrmap_img = irrmap.updateMask(mask).remap([0, 1, 2], [1, 0, 0])

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                data_url = irrmap_img.getDownloadURL({'name': data_name,
                                                      'crs': 'EPSG:4269',  # NAD83
                                                      'scale': 30,  # in meter
                                                      'region': gee_extent,
                                                      'format': 'GEO_TIFF'})

                r = requests.get(data_url, allow_redirects=True)

                output_raster = os.path.join(download_dir, f'{data_name}_{year}_{grid_sr}.tif')
                print('Downloading', output_raster, '.....')

                # save data to local file path
                open(output_raster, 'wb').write(r.content)

                # This is a check block to see if downloaded datasets are OK
                # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
                # Re-downloading the same data might not have that error
                try:
                    arr = read_raster_arr_object(output_raster, get_file=False)

                except:
                    print(f'Downloaded data corrupted. Re-downloading {output_raster}.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(output_raster, 'wb').write(r.content)


def download_rainfed_cropland_30m(data_name, years, grid_shape, download_dir):
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    if 'KS' in data_name:  # for GMD, kansas
        # CDL dataset information required for downloading from GEE
        cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, _, _ = get_gee_dict('USDA_CDL')

        # Extracting dataset information required for downloading from GEE
        lanid_asset, _, _, _, _, _, _, _ = get_gee_dict('LANID')
        lanid_data_band_dict = {2000: 'lanid_2000', 2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003',
                                2004: 'lanid_2004', 2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007',
                                2008: 'lanid_2008', 2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011',
                                2012: 'lanid_2012', 2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015',
                                2016: 'lanid_2016', 2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019',
                                2020: 'lanid_2020'}

        # Extracting AIM-HPA dataset information (saved as an asset) from GEE
        AimHpa_asset, _, _, _, _, _, _, _ = get_gee_dict('AIM-HPA')
        aim_hpa_data_band_dict = {2000: 'aim_2000', 2001: 'aim_2001', 2002: 'aim_2002', 2003: 'aim_2003',
                                  2004: 'aim_2004', 2005: 'aim_2005', 2006: 'aim_2006', 2007: 'aim_2007',
                                  2008: 'aim_2008', 2009: 'aim_2009', 2010: 'aim_2010', 2011: 'aim_2011',
                                  2012: 'aim_2012', 2013: 'aim_2013', 2014: 'aim_2014', 2015: 'aim_2015',
                                  2016: 'aim_2016', 2017: 'aim_2017', 2018: 'aim_2018', 2019: 'aim_2019',
                                  2020: 'aim_2020'}

        for year in years:  # first loop for years
            print('********')
            print(f'Getting Rainfed Cropland data for year={year}.....')

            # # USDA CDL data for the year
            cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
                .first().select(cdl_band)

            #  List of non-crop pixels
            noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88,
                                    111, 112, 121, 122, 123,
                                    124, 131, 141, 142, 143,
                                    152, 190, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class

            # Filtering out non-crop pixels
            cdl_cropland = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

            # ######### Merging irrigated field dataset from LANID and AIM-HPA
            # # LANID data for the year
            # In LANID dataset irrigated fields are assigned as 1
            lanid_band = lanid_data_band_dict[year]
            irr_lanid = ee.Image(lanid_asset).select(lanid_band)
            irr_lanid = irr_lanid.eq(1).unmask()  # unmask() assigns zero value to no data pixels

            # # AIM-HPA data for the year
            # In AIM-HPA dataset irrigated fields are assigned as 1
            hpa_band = aim_hpa_data_band_dict[year]
            irr_hpa = ee.Image(AimHpa_asset).select(hpa_band)
            irr_hpa = irr_hpa.eq(1).unmask()  # unmask() assigns zero value to no data pixels

            # # Joining LANID and AIM-HPA
            # The equal operation (with 0) sets non-irrigated locations to 1 and irrigated to 0
            # Remapping the irrigated 0 values to 1 (others to 0)
            irr_total = irr_lanid.add(irr_hpa)
            irr_total = irr_total.eq(0).remap([0], [1]).unmask()

            # Creating Rainfed dataset by combining LANID + AIM-HPA and USDA CDL data
            # Extracting rainfed (non-irrigated) croplands
            # Converting 0 values (non-irrigated) to 1, and 1 (irrigated) to 0
            # Then multiplying with CDL cropland data (reclassified to 1 for croplands and 0 for non-croplands).
            # The remaining pixels will be rainfed croplands. Masking it again to remove the 0 values
            irr_total_reversed = irr_total.remap([0, 1], [1, 0])
            rainfed_cropland = cdl_cropland.multiply(irr_total_reversed)
            mask = rainfed_cropland.eq(1)
            rainfed_cropland = rainfed_cropland.updateMask(mask)

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                data_url = rainfed_cropland.getDownloadURL({'name': data_name,
                                                            'crs': 'EPSG:4269',  # NAD83
                                                            'scale': 30,  # in meter
                                                            'region': gee_extent,
                                                            'format': 'GEO_TIFF'})

                r = requests.get(data_url, allow_redirects=True)

                output_raster = os.path.join(download_dir, f'{data_name}_{year}_{grid_sr}.tif')
                print('Downloading', output_raster, '.....')

                # save data to local file path
                open(output_raster, 'wb').write(r.content)

                # This is a check block to see if downloaded datasets are OK
                # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
                # Re-downloading the same data might not have that error
                try:
                    arr = read_raster_arr_object(output_raster, get_file=False)

                except:
                    print(f'Downloaded data corrupted. Re-downloading {output_raster}.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(output_raster, 'wb').write(r.content)

    elif 'CA' in data_name:  # for Central Valley, CA
        # CDL dataset information required for downloading from GEE
        cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, _, _ = get_gee_dict('USDA_CDL')

        # Extracting dataset information required for downloading from GEE
        irrmapper_asset, irrmapper_band, multiply_scale, reducer, _, _, _, _= get_gee_dict('IrrMapper')

        for year in years:  # first loop for years
            print('********')
            print(f'Getting Irrigated Cropland data for year={year}.....')

            # # USDA CDL data for the year
            cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
                .first().select(cdl_band)

            #  List of non-crop pixels
            noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88,
                                    111, 112, 121, 122, 123,
                                    124, 131, 141, 142, 143,
                                    152, 190, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class

            # Filtering out non-crop pixels
            cdl_cropland = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

            # Filtering data for the year range and reducing data
            irrmap = ee.ImageCollection(irrmapper_asset).filter(ee.Filter.calendarRange(year, year, 'year'))\
                                        .select(irrmapper_band).max().toFloat()

            # In IrrMapper dataset irrigated fields are assigned as 0
            # Converting the irrigated values to 1 and setting others as 0
            irrmap_img = irrmap.eq(0).unmask()

            # Extracting rainfed (non-irrigated) croplands
            # Converting 0 values (non-irrigated) to 1 then multiplying with
            # CDL cropland data (reclassified to 1 for croplands and 0 for non-croplands).
            # The remaining pixels will be rainfed croplands. Masking it again to remove the 0 values
            irrmapper_reversed = irrmap_img.remap([0, 1], [1, 0])
            rainfed_cropland = cdl_cropland.multiply(irrmapper_reversed)
            mask = rainfed_cropland.eq(1)
            rainfed_cropland = rainfed_cropland.updateMask(mask)

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                data_url = rainfed_cropland.getDownloadURL({'name': data_name,
                                                            'crs': 'EPSG:4269',  # NAD83
                                                            'scale': 30,  # in meter
                                                            'region': gee_extent,
                                                            'format': 'GEO_TIFF'})

                r = requests.get(data_url, allow_redirects=True)

                output_raster = os.path.join(download_dir, f'{data_name}_{year}_{grid_sr}.tif')
                print('Downloading', output_raster, '.....')

                # save data to local file path
                open(output_raster, 'wb').write(r.content)

                # This is a check block to see if downloaded datasets are OK
                # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
                # Re-downloading the same data might not have that error
                try:
                    arr = read_raster_arr_object(output_raster, get_file=False)

                except:
                    print(f'Downloaded data corrupted. Re-downloading {output_raster}.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(output_raster, 'wb').write(r.content)


def download_cdl_30m(data_name, years, grid_shape, download_dir):
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    # CDL dataset information required for downloading from GEE
    cdl_data, cdl_band, cdl_multiply_scale, cdl_reducer, _, _, _, _ = get_gee_dict('USDA_CDL')

    for year in years:  # first loop for years
        print('********')
        print(f'Getting Rainfed Cropland data for year={year}.....')

        # # USDA CDL data for the year
        cdl_img = ee.ImageCollection(cdl_data).filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first().select(cdl_band)

        #  List of non-crop pixels
        noncrop_list = ee.List([0, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88,
                                111, 112, 121, 122, 123,
                                124, 131, 141, 142, 143,
                                152, 190, 195])  # 0 is no data value, keeping 176 (pasture) in cropland class

        # Filtering out non-crop pixels
        cdl_cropland_filtered = cdl_img.remap(noncrop_list, ee.List.repeat(0, noncrop_list.size()), 1)

        # Multiplying with the cdl_cropland filter to keep only cropland classes of interest
        cdl_img = cdl_img.multiply(cdl_cropland_filtered)

        for i in range(len(grid_no)):  # third loop for grids
            # converting grid geometry info to a GEE extent
            grid_sr = grid_no[i]
            roi = grid_geometry[i].bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            data_url = cdl_img.getDownloadURL({'name': data_name,
                                                        'crs': 'EPSG:4269',  # NAD83
                                                        'scale': 30,  # in meter
                                                        'region': gee_extent,
                                                        'format': 'GEO_TIFF'})

            r = requests.get(data_url, allow_redirects=True)

            output_raster = os.path.join(download_dir, f'{data_name}_{year}_{grid_sr}.tif')
            print('Downloading', output_raster, '.....')

            # save data to local file path
            open(output_raster, 'wb').write(r.content)

            # This is a check block to see if downloaded datasets are OK
            # sometimes a particular grid's data is corrupted but it's completely random, not sure why it happens.
            # Re-downloading the same data might not have that error
            try:
                arr = read_raster_arr_object(output_raster, get_file=False)

            except:
                print(f'Downloaded data corrupted. Re-downloading {output_raster}.....')
                r = requests.get(data_url, allow_redirects=True)
                open(output_raster, 'wb').write(r.content)


# GEE data download grids
KS_30m_grids = '../../Data_main/Data_Drivers_of_ET/ROIs/GEE_shapefiles/KS_grids_30m.shp'
CA_30m_grids = '../../Data_main/Data_Drivers_of_ET/ROIs/GEE_shapefiles/CA_grids_30m.shp'

# Main download directories
KS_download_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS'
CA_download_dir = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/CA'


years_to_download_data_for = [2016, 2017, 2018, 2019, 2020]

# Downloading data for Kansas ROI GMD4 (County: Cheyenne, Rawlins, Sherman, Decatur, Thomas, Sheridan, Graham, Logan, Gove)
# download_openET_30m(data_name='OpenET_ensemble', years=years_to_download_data_for, months=(1, 12),
#                     grid_shape=KS_30m_grids, download_dir=KS_download_dir)
#
# download_irrigated_cropland_30m(data_name='Irrigated_KS', years=years_to_download_data_for,
#                                 grid_shape=KS_30m_grids, download_dir=KS_download_dir)
#
# download_rainfed_cropland_30m(data_name='Rainfed_KS', years=years_to_download_data_for,
#                                 grid_shape=KS_30m_grids, download_dir=KS_download_dir)
#
#
# download_cdl_30m(data_name='CDL_KS', years=years_to_download_data_for,
#                  grid_shape=KS_30m_grids, download_dir=KS_download_dir)
#
#
# # Downloading data for California ROI (County: Tehama, Glenn, Butte, Colusa, Sutter, Yuba)
# download_openET_30m(data_name='OpenET_ensemble', years=years_to_download_data_for, months=(1, 12),
#                     grid_shape=CA_30m_grids, download_dir=CA_download_dir)
#
# download_irrigated_cropland_30m(data_name='Irrigated_CA', years=years_to_download_data_for,
#                                 grid_shape=CA_30m_grids, download_dir=CA_download_dir)
#
# download_rainfed_cropland_30m(data_name='Rainfed_CA', years=years_to_download_data_for,
#                                 grid_shape=CA_30m_grids, download_dir=CA_download_dir)
#
# download_cdl_30m(data_name='CDL_CA', years=years_to_download_data_for,
#                  grid_shape=CA_30m_grids, download_dir=CA_download_dir)
#

# Downloading data for Kansas ROI GMD2
KS_30m_grids_gmd2 = '../../Data_main/Data_Drivers_of_ET/ROIs/GEE_shapefiles/KS_grids_30m_GMD2.shp'
KS_download_dir2 = '../../Data_main/Data_Drivers_of_ET/Raster_data/30m/KS_2'


# download_openET_30m(data_name='OpenET_ensemble', years=years_to_download_data_for, months=(1, 12),
#                     grid_shape=KS_30m_grids_gmd2, download_dir=KS_download_dir2)
#
# download_irrigated_cropland_30m(data_name='Irrigated_KS', years=years_to_download_data_for,
#                                 grid_shape=KS_30m_grids_gmd2, download_dir=KS_download_dir2)
#
# download_rainfed_cropland_30m(data_name='Rainfed_KS', years=years_to_download_data_for,
#                                 grid_shape=KS_30m_grids_gmd2, download_dir=KS_download_dir2)
#
#
# download_cdl_30m(data_name='CDL_KS', years=years_to_download_data_for,
#                  grid_shape=KS_30m_grids_gmd2, download_dir=KS_download_dir2)