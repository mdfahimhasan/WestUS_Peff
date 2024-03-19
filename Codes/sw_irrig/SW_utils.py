import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from rasterstats import zonal_stats

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs, copy_file
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'


def create_canal_coverage_raster(years_list, canal_shapefile, irrig_cropET_grow_season_dir,
                                 canal_coverage_output_dir, irrigated_CropET_with_canal_coverage_dir,
                                 ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Create canal coverage rasters for all year_list. Canal coverage is only assigned to pixels that fall
    within canal coverage shapefile (or touches canal coverage shapefile) and there is irrigated cropET in that pixel
    for that year.

    :param years_list: A list of year_list to process data for.
    :param canal_shapefile: Filepath of canal coverage shapefile.
    :param irrig_cropET_grow_season_dir: Directory path of irrigated cropET growing season rasters.
    :param canal_coverage_output_dir: Output directory path to save canal coverage rasters.
    :param irrigated_CropET_with_canal_coverage_dir: Output directory path to save irrigated cropET growing season
                                                     rasters overlaid by respective year's canal coverage raster.
    :param ref_raster: Filepath of reference raster.
    :param resolution: Model resolution.
    :param skip_processing: Set to True to skip this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([canal_coverage_output_dir, irrigated_CropET_with_canal_coverage_dir])

        # creating an overall canal coverage raster, where all pixels that have canal coverage have value 1
        canal_raster = shapefile_to_raster(input_shape=canal_shapefile, output_dir=canal_coverage_output_dir,
                                           raster_name='canal_coverage.tif', burnvalue=1, use_attr=False, add=None,
                                           ref_raster=ref_raster, resolution=resolution, alltouched=True)
        canal_arr = read_raster_arr_object(canal_raster, get_file=None)

        # now we will loop over irrigated cropET for each growing season and only assign canal
        # coverage to pixels that have some values of irrigated cropET.
        # Also saving irrigated cropET where we have canal coverage
        for year in years_list:
            print(f'Creating canal coverage (with growing season irrigated cropET) raster for {year}...')

            irrig_cropET = glob(os.path.join(irrig_cropET_grow_season_dir, f'*{year}*.tif'))[0]
            irrig_cropET_arr, file = read_raster_arr_object(irrig_cropET)

            # final canal coverage raster
            canal_coverage_arr = np.where((canal_arr == 1) & (~np.isnan(irrig_cropET_arr)), 1, -9999)

            output_raster = os.path.join(canal_coverage_output_dir, f'canal_coverage_{year}.tif')
            write_array_to_raster(raster_arr=canal_coverage_arr, raster_file=file, transform=file.transform,
                                  output_path=output_raster)

            # irrigated cropET overlaid on canal coverage raster
            cropET_canal_cover_arr = np.where((canal_arr == 1) & (~np.isnan(irrig_cropET_arr)), irrig_cropET_arr, -9999)

            output_raster = os.path.join(irrigated_CropET_with_canal_coverage_dir,
                                         f'irrigated_cropET_with_canals_{year}.tif')
            write_array_to_raster(raster_arr=cropET_canal_cover_arr, raster_file=file, transform=file.transform,
                                  output_path=output_raster)
    else:
        pass


def estimate_sw_mm_HUC12(years_list, HUC12_input_shapefile, irrigated_CropET_with_canal_coverage_dir,
                         HUC12_output_shapefile, skip_precessing=False):
    """
    Estimate number of pixels that falls in canal coverage and total irrigated cropET in those pixels for each
    HUC12 watershed. Also, calculates growing season SW irrigation in mm (area averaged, considers canal covered
    irrigated cropET pixels, growing season) for all HUC12s.

    :param years_list: A list of year_list to process data for.
    :param HUC12_input_shapefile: Filepath of WestUS HUC12 shapefile.
    :param irrigated_CropET_with_canal_coverage_dir: Directory path of irrigated cropET growing season
                                                     rasters (overlaid with canal coverage raster).
    :param HUC12_output_shapefile: Filepath of HUC12 output shapefile with total canal covered pixel and total
                                   irrigated cropET data along with SW irrigation data in mm.
    :param skip_precessing: Set to True to skip this step.

    :return: None.
    """
    if not skip_precessing:
        HUC12_gdf = gpd.read_file(HUC12_input_shapefile)

        # an empty dictionary of lists to store the results
        results = {'huc12': [],
                   'pixel_2016': [], 'ET2016_mm': [],
                   'pixel_2017': [], 'ET2017_mm': [],
                   'pixel_2018': [], 'ET2018_mm': [],
                   'pixel_2019': [], 'ET2019_mm': [],
                   'pixel_2020': [], 'ET2020_mm': []}

        for year in years_list:  # looping through growing season irrigated cropET data to extract watershed/HUC12
            # level information
            print(f'Extracting total irrigated cropET and number of pixels stats in HUC12s for {year}...')

            # irrigated cropET growing season with canal coverage for that year
            irrig_cropET_with_canal = glob(os.path.join(irrigated_CropET_with_canal_coverage_dir, f'*{year}*.tif'))[0]

            for idx, row in HUC12_gdf.iterrows():  # looping through each HUC12 watershed and collecting data
                huc12_geom = row['geometry']

                # performing zonal statistics to collect data
                ET_stat = zonal_stats(huc12_geom, irrig_cropET_with_canal, stats='sum')
                pixel_stat = zonal_stats(huc12_geom, irrig_cropET_with_canal, stats='count')

                # appending the result to the empty lists
                results[f'ET{year}_mm'].append(ET_stat[0]['sum'])   # sum of total irrigated crop ET in the HUC12
                results[f'pixel_{year}'].append(pixel_stat[0]['count'])  # count of total canal-covered irrigated crop ET pixels in the HUC12

                if year == years_list[0]:  # will populate HUC12 no. list only once. Otherwise it will keep appending for each year
                    results['huc12'].append(row['huc12'])
                else:
                    pass

        # converting the results into a dataframe
        results_df = pd.DataFrame(results)

        # merging results dataframe with the HUC12 geodataframe
        HUC12_gdf_merged = HUC12_gdf.merge(results_df, on='huc12')
        print('Columns in HUC12 shapefile:', '\n', HUC12_gdf_merged.columns)

        # # converting MDG to mm
        # area of a pixel
        area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2

        for year in years_list:
            sw_mm3 = HUC12_gdf_merged[f'{year}'] * 3785411784000 * 214  # conversion from MGD to mm3/grow season; 214 days from April-October
            area_irrig_pixels = area_mm2_single_pixel * HUC12_gdf_merged[f'pixel_{year}']   # unit mm2
            HUC12_gdf_merged[f'sw_{year}_mm'] = sw_mm3 / area_irrig_pixels   # unit mm/grow season

        # saving finalized shapefile
        HUC12_gdf_merged.to_file(HUC12_output_shapefile)

    else:
        pass


def distribute_SW_irrigation_to_pixels(years_list, HUC12_shapefile, irrigated_CropET_with_canal_coverage_dir,
                                       sw_dist_outdir, ref_raster=WestUS_raster,
                                       resolution=model_res, skip_processing=False):
    """
    Distribute HUC12 level surface water irrigation (unit mm/growing season) to irrigated pixels that have 
    canal coverage (within 2 km buffer of canal).
    
    :param years_list:  A list of year_list to process data for.
    :param HUC12_shapefile: Filepath of HUC12 shapefile with total canal covered pixels, total
                             irrigated cropET data, and SW irrigation data in mm.
    :param irrigated_CropET_with_canal_coverage_dir: Directory path of irrigated cropET growing season
                                                    rasters (overlaid with canal coverage raster).
    :param sw_dist_outdir: Output directory to save sw distributed rasters.
    :param ref_raster: Filepath of Western US reference raster.
    :param resolution: Model resolution.
    :param skip_processing: Set to True to skip this step.

    :return: None.
    """
    if not skip_processing:
        total_cropET_SW_dir = os.path.join(sw_dist_outdir, 'total_cropET_SW')
        makedirs([sw_dist_outdir, total_cropET_SW_dir])

        # replacing null values with 0 in the HUC12 shapefile and saving it before distributing
        huc12_gdf = gpd.read_file(HUC12_shapefile)
        huc12_gdf = huc12_gdf.replace([np.inf, np.nan], 0)

        HUC12_processed = os.path.join(total_cropET_SW_dir, 'HUC12_processed.shp')
        huc12_gdf.to_file(HUC12_processed)

        # reference raster
        ref_arr, ref_file = read_raster_arr_object(ref_raster)

        for year in years_list:
            print(f'distributing surface water irrigation to pixels for {year}...')

            # getting growing season irrigated cropET raster, which has canal coverage overlaid (not HUC12 sum)
            irrig_cropET_Huc12_tot = glob(os.path.join(irrigated_CropET_with_canal_coverage_dir, f'*{year}*.tif'))[0]

            # converting total irrigated cropET of HUC12 to raster (HUC12 sum)
            total_irrig_cropET_ras = f'total_irrig_cropET_{year}.tif'
            attr_to_use = f'ET{year}_mm'

            total_irrig_cropET = shapefile_to_raster(input_shape=HUC12_processed, output_dir=total_cropET_SW_dir,
                                                     raster_name=total_irrig_cropET_ras,
                                                     use_attr=True, attribute=attr_to_use,
                                                     ref_raster=ref_raster, resolution=resolution)

            # converting total SW irrigation of HUC12 to raster
            total_sw_irrig_ras = f'total_SW_irrig_{year}.tif'
            attr_to_use = f'sw_{year}_mm'

            total_sw_irrig = shapefile_to_raster(input_shape=HUC12_processed, output_dir=total_cropET_SW_dir,
                                                 raster_name=total_sw_irrig_ras,
                                                 use_attr=True, attribute=attr_to_use,
                                                 ref_raster=ref_raster, resolution=resolution)

            # array operation to distribute total sw irrigation in a HUC12 to
            # all its irrigated pixels that have canal coverage
            irrig_cropET_canal_cover_arr = read_raster_arr_object(irrig_cropET_Huc12_tot, get_file=False)
            total_irrig_cropET_arr = read_raster_arr_object(total_irrig_cropET, get_file=False)
            sw_irrig_arr = read_raster_arr_object(total_sw_irrig, get_file=False)

            # the total sw irrigation will be distributed to a pixel based on its ratio
            # of irrigated cropET in a pixel/total irrigated cropET in the HUC12
            sw_dist_arr = np.where((sw_irrig_arr != 0) | (irrig_cropET_canal_cover_arr != 0) | (total_irrig_cropET_arr != 0),
                                    sw_irrig_arr * (irrig_cropET_canal_cover_arr/total_irrig_cropET_arr), -9999)

            sw_initial_output_dir = os.path.join(sw_dist_outdir, 'SW_dist_initial')
            makedirs([sw_initial_output_dir])
            sw_dist_raster_initial = os.path.join(sw_initial_output_dir, f'sw_irrigation_{year}_initial.tif')
            write_array_to_raster(raster_arr=sw_dist_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=sw_dist_raster_initial)

            # assigning zero values to pixels with no surface water irrigation.
            # this is specially an important step for regions with groundwater pumping
            # but no surface irrigation (during calculation of netGW)
            # regions out of Western US (sea and others) are assigned no data value
            sw_dist_arr = np.where(~np.isnan(sw_dist_arr), sw_dist_arr, 0)
            sw_dist_arr = np.where(ref_arr == 0, sw_dist_arr, ref_arr)  # assigning no data value

            sw_dist_raster = os.path.join(sw_dist_outdir, f'sw_irrigation_{year}.tif')
            write_array_to_raster(raster_arr=sw_dist_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=sw_dist_raster)

    else:
        pass
