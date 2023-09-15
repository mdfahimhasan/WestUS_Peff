import os
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd

from Codes.utils.vector_ops import add_attr_to_county_fromCSV
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, shapefile_to_raster
from Codes.utils.stats_ops import calc_outlier_ranges_MAD
from Codes.utils.system_ops import makedirs

no_data_value = -9999
model_res = 0.02000000000000000736  # in deg, ~2.22 km
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def create_canal_coverage_raster(canal_shapefile='../../Data_main/shapefiles/Surface_water_shapes/canals_buffered_2km_epsg_4269.shp',
                                 output_dir='../../Data_main/Raster_data/Canal_coverage', ref_raster=WestUS_raster):
    makedirs([output_dir])

    shapefile_to_raster(input_shape=canal_shapefile, output_dir=output_dir, raster_name='canal_coverage.tif',
                        burnvalue=1, use_attr=False, attribute="", add=None,
                        ref_raster=ref_raster, resolution=model_res, alltouched=False)

    canal_coverage_raster = os.path.join(output_dir, 'canal_coverage.tif')

    return canal_coverage_raster


def compile_USGS_WaterUse_data(yearlist, usgs_data_dir='../../Data_main/USGS_water_use_data',
                               county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county_projected.shp',
                               irrigated_data_dir='../../Data_main/Raster_data/Irrigated_agriculture/WestUS_Processed_irrigated_lands',
                               county_id_data='../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif',
                               canal_coverage_data = '../../Data_main/Raster_data/Canal_coverage/canal_coverage.tif',
                               output_dir='../../Data_main/USGS_water_use_data',
                               skip_compiling=False, ref_raster=WestUS_raster):
    """
    Compile county-wide water use data for Western US from USGS dataset.

    :param data_dir: Directory path of yearly groundwater data excels.
    :param search_by: Search pattern for yearly data excels. Default set to '*201[0-5]*.xlsx' for selecting 2010 and
                      2015 data only.
    :param county_shape: File path of Western US county shapefile location.
    :param output_dir: Directory path to save output csv and shapefile data.
    :param skip_compiling: Set to True to skip compiling water use data. Default set to False to compile data.
    :param ref_raster: Model reference raster filepath.

    :return: Compiled USGS pumping data csv.
    """
    if not skip_compiling:
        import warnings
        warnings.simplefilter('ignore')  # adding to remove warning from excel opening

        # Loading required datasets
        ref_arr, _ = read_raster_arr_object(ref_raster)
        county_df = gpd.read_file(county_shape)
        countyID_arr, county_file = read_raster_arr_object(county_id_data)

        canal_cover_arr = read_raster_arr_object(canal_coverage_data, get_file=False)

        # initiating an empty dataframe to store yearly county-level water use data
        county_wateruse = pd.DataFrame()

        # Dataframe to track SW Use Per pixels values used to detect outliers and median values per year
        outlier_detection_df = pd.DataFrame()

        # Dictionary to store individual year's dataframe temporarily. Will be used in missing data imputation for 2005
        yearly_data_dict = {}

        # Looping through each year and storing data in the county_wateruse dataframe
        for year in sorted(yearlist, reverse=True):
            print(f'Compiling USGS Water Use Data for {year}...')

            # Counting Irrigated crop pixels in each county
            irrig_data = glob(os.path.join(irrigated_data_dir, f'*{year}*.tif'))[0]
            irrig_arr = read_raster_arr_object(irrig_data, get_file=False)

            # Counting how many pixels in each county has SW irrigated cropland pixels with canal coverage
            unique, count = np.unique(countyID_arr[(irrig_arr == 1) &
                                                   (canal_cover_arr == 1) &
                                                   (~np.isnan(countyID_arr))], return_counts=True)
            count_dict = dict(zip(unique, count))

            # Opening USGS yearly water use data excel
            wateruse_data = glob(os.path.join(usgs_data_dir, f'*{year}*.xlsx'))[0]
            wateruse_df = pd.read_excel(wateruse_data, sheet_name='CountyData', engine='openpyxl')

            # Filtering required datasets only. These water use datasets are county level total values in Mgal/day
            # # IC-WGWFr: Irrig-crop GW, IC-WSWFr: Irrig-crop SW, IC-WFrTo: Irrig-crop total
            # # TO-WGWFr: Total GW withdrawal, TO-WSWFr: Total SW withdrawal, TO-WFrTo: Total freshwater withdrawal
            wateruse_df = wateruse_df[['COUNTY', 'STATE', 'COUNTYFIPS', 'FIPS', 'YEAR',
                                       'IC-WGWFr', 'IC-WSWFr', 'IC-WFrTo',
                                       'TO-WGWFr', 'TO-WSWFr', 'TO-WFrTo']]

            wateruse_df[['IC-WGWFr', 'IC-WSWFr', 'IC-WFrTo', 'TO-WGWFr', 'TO-WSWFr', 'TO-WFrTo']] = \
                wateruse_df[['IC-WGWFr', 'IC-WSWFr', 'IC-WFrTo', 'TO-WGWFr', 'TO-WSWFr', 'TO-WFrTo']].\
                    apply(pd.to_numeric, errors='coerce')

            joined_df = county_df.merge(wateruse_df, left_on='fips', right_on='FIPS', how='inner')

            # adding irrigated cropland pixel counts to the dataframe
            joined_df['irr_pixels'] = None
            joined_df['irr_pixels'] = joined_df['fips'].map(count_dict)

            # converting water use from Mgal/day to mm/year for the total county
            area_single_pixel = (2.22 * 2.22) * (1000 * 1000)  # area of a pixel in m2
            joined_df['irr_gw'] = 1000 * (1e6 * joined_df['IC-WGWFr'] * 0.00378541 * 365 /
                                                       (joined_df['irr_pixels'] * area_single_pixel))
            joined_df['irr_sw'] = 1000 * (1e6 * joined_df['IC-WSWFr'] * 0.00378541 * 365 /
                                                       (joined_df['irr_pixels'] * area_single_pixel))
            joined_df['irr_tot'] = 1000 * (1e6 * joined_df['IC-WFrTo'] * 0.00378541 * 365 /
                                                          (joined_df['irr_pixels'] * area_single_pixel))

            # # # This block is for imputing missing data in 2005
            yearly_data_dict[year] = [joined_df]  # storing df for imputing missing data in 2005

            if year == 2005:
                # gathering total county sw use data for 2005 and 2010
                irr_sw_2010 = yearly_data_dict[2010][0]['irr_sw'].tolist()
                irr_sw_2005 = yearly_data_dict[2005][0]['irr_sw'].tolist()

                # creating an empty list to store 2005 total county sw use data along with imputed data from 2010
                irr_sw_2005_imputed = []

                for data_2010, data_2005 in zip(irr_sw_2010, irr_sw_2005):
                    if ~np.isnan(data_2005):
                        irr_sw_2005_imputed.append(data_2005)
                    elif np.isnan(data_2005) & ~np.isnan(data_2010):
                        irr_sw_2005_imputed.append(data_2010)
                    else:
                        irr_sw_2005_imputed.append(np.nan)

                joined_df['irr_sw'] = irr_sw_2005_imputed
            # # #

            # # # read me
            # there are outliers in area averaged sw-irrigation data ('irr_sw') for each county.
            # The outliers are difficult to remove as 'irr_sw' is not our final per pixel sw use product.
            # Further outlier might be created if such values of 'irr_sw' exists in counties  where number of
            # irrigated pixels are also low, because that will create high values of SW use per pixel ('sw_pix').
            # So, we will first calculate SW use per pixel ('sw_pix') and remove outlier using these values.
            # The outlier removed product is 'sw_pix_2' where outlier have been imputed by median values.
            # But the 'sw_pix_2' considers that the sw use value of a country is equally distributed in all
            # sw-irrigated pixels. In our methodology, we will not assume equal distribution,
            # rather total SW use of a county will be distributed based on crop ET percentage in each pixel.
            # Therefore, the 'sw_pix_2' is not the final sw distribution product rather a value created by
            # removing outliers. We will multiply 'sw_pix_2' with the 'number of sw irrigated pixels' to recreate a
            # outlier removed version of sw irrigation in a county 'irrSWfinal'.
            # The 'irrSWfinal' value will be later distributed using crop ET percentage per pixel.

            # Estimating sw use per irrigated pixel (with canal coverage)
            joined_df['sw_pix'] = joined_df['irr_sw'] / joined_df['irr_pixels']  # have outliers
            joined_df['Year'] = [year for i in range(len(joined_df))]

            # Calculating outlier ranges and median value
            # Saving outlier detection info to outlier_detection_df dataframe
            increase_upper_range_by = 0
            lower_range, upper_range, median = \
                calc_outlier_ranges_MAD(joined_df['sw_pix'],
                                        increase_upper_range_by=increase_upper_range_by)

            yearly_outlier_detection_df = pd.DataFrame({'Year': [year],
                                                        '25th percentile SW use/pixel': [lower_range],
                                                        'Median SW use/pixel': [median],
                                                        '75th percentile SW use/pixel': [upper_range + increase_upper_range_by],
                                                        'Adjusted Upper range SW use/pixel': [upper_range]})

            outlier_detection_df = pd.concat([outlier_detection_df, yearly_outlier_detection_df])

            # Outlier Removal : Removing outliers from 'sw_pix'
            # replacing non-nan outlier values with median
            sw_use_outlier_removed = []

            for i in joined_df['sw_pix'].tolist():
                if lower_range <= i <= upper_range:
                    sw_use_outlier_removed.append(i)
                elif np.isnan(i):
                    sw_use_outlier_removed.append(i)
                else:
                    sw_use_outlier_removed.append(median)

            joined_df['sw_pix_2'] = sw_use_outlier_removed
            joined_df['irrSWfinal'] = joined_df['sw_pix_2'] * joined_df['irr_pixels']

            # 'Sw_pix' is calculated sw use per sw-irrigated pixel in mm/yr with outliers.
            # 'Sw_pix_2' is sw use per sw irrigated pixel in mm/yr with outliers removed.
            # 'irrSWfinal' is the outlier removed total sw use of a country
            # in 'irrSWfinal' outlier values have been imputed (median of 'Sw_pix_2' + sw-irrigated pixel in county)
            joined_df = joined_df[['COUNTY', 'STATE', 'fips', 'Year',
                                   'IC-WGWFr', 'IC-WSWFr', 'IC-WFrTo',
                                   'irr_pixels',
                                   'irr_gw', 'irr_sw', 'irr_tot',
                                   'sw_pix', 'sw_pix_2', 'irrSWfinal']]

            # Converting to shapefile
            output_shapefile_path = os.path.join(output_dir, f'WestUS_county_WaterUse_{year}.shp')
            add_attr_to_county_fromCSV(
                input_shapefile='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county.shp',
                attr_csv_df_data=joined_df,
                output_shapefile=output_shapefile_path, year_filter=year,
                columns_to_keep=None)

            # Adding each year's data to a final dataframe
            county_wateruse = pd.concat([county_wateruse, joined_df])

        # Saving as csv
        wateruse_compiled_csv = os.path.join(output_dir, 'WestUS_county_WaterUse.csv')
        county_wateruse.to_csv(wateruse_compiled_csv, index=False)

        outlier_detection_df.to_csv(os.path.join(output_dir, 'outlier_detect_range.csv'), index=False)

        print('USGS Water Use Data Compiled\n')

    else:
        print('Loading USGS Water Use Data...\n')
        wateruse_compiled_csv = os.path.join(output_dir, 'WestUS_county_WaterUse.csv')

    return wateruse_compiled_csv


compile_USGS_WaterUse_data(yearlist=[2005, 2010, 2015], skip_compiling=False)