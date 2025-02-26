import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from osgeo import gdal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, clip_resample_reproject_raster

no_data_value = -9999
model_res = 2000  # in m
AZ_shape = '../../Data_main/AZ_files/ref_files/AZ.shp'
AZ_raster = '../../Data_main/AZ_files/ref_files/AZ_ref_raster.tif'


def create_monthly_dataframes_for_eff_precip_prediction(years_list, month_range,
                                                        monthly_data_path_dict, yearly_data_path_dict,
                                                        static_data_path_dict, datasets_to_include, output_dir,
                                                        skip_processing=False):
    """
    Create monthly dataframes of predictors to generate monthly effective prediction.

    :param years_list: A list of years_list for which data to include in the dataframe.
    :param month_range: A tuple of start and end month for which data to filter. Set to None if there is no monthly dataset.
    :param monthly_data_path_dict: A dictionary with monthly variables' names as keys and their paths as values.
    :param yearly_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param static_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Set to None if there is no yearly dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this dataframe creation process.

    :return: None
    """
    if not skip_processing:
        makedirs([output_dir])

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

        for year in years_list:  # 1st loop controlling years_list
            for month in month_list:  # 2nd loop controlling months

                if year == 1984 and month in range(1, 10):  # skipping dataframe creation for 1984 January-September
                    continue

                else:
                    print(f'creating dataframe for prediction - year={year}, month={month}...')

                    variable_dict = {}

                    # reading monthly data and storing it in a dictionary
                    for var in monthly_data_path_dict.keys():
                        if var in datasets_to_include:

                            if var == 'GRIDMET_Precip':  # for including monthly and lagged monthly GRIDMET_precip in the dataframe
                                current_precip_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]

                                current_month_date = datetime(year, month, 1)

                                # Collect previous month's precip data
                                prev_month_date = current_month_date - timedelta(30)
                                prev_2_month_date = current_month_date - timedelta(60)

                                prev_month_precip_data = glob(os.path.join(monthly_data_path_dict[var],
                                                                           f'*{prev_month_date.year}_{prev_month_date.month}.tif*'))[0]
                                prev_2_month_precip_data = glob(os.path.join(monthly_data_path_dict[var],
                                                                             f'*{prev_2_month_date.year}_{prev_2_month_date.month}.tif*'))[0]

                                # reading datasets
                                current_precip_arr = read_raster_arr_object(current_precip_data, get_file=False).flatten()

                                prev_month_precip_arr = read_raster_arr_object(prev_month_precip_data, get_file=False).flatten()
                                prev_2_month_precip_arr = read_raster_arr_object(prev_2_month_precip_data, get_file=False).flatten()

                                current_precip_arr[np.isnan(current_precip_arr)] = 0  # setting nan-position values with 0
                                prev_month_precip_arr[np.isnan(prev_month_precip_arr)] = 0  # setting nan-position values with 0
                                prev_2_month_precip_arr [np.isnan(prev_2_month_precip_arr)] = 0  # setting nan-position values with 0

                                variable_dict[var] = list(current_precip_arr)
                                variable_dict['month'] = [int(month)] * len(current_precip_arr)
                                variable_dict['GRIDMET_Precip_1_lag'] = list(prev_month_precip_arr)
                                variable_dict['GRIDMET_Precip_2_lag'] = list(prev_2_month_precip_arr)

                            else:
                                monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]
                                data_arr = read_raster_arr_object(monthly_data, get_file=False).flatten()

                                data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                                variable_dict[var] = list(data_arr)
                                variable_dict['month'] = [int(month)] * len(data_arr)

                # reading yearly data and storing it in a dictionary
                if yearly_data_path_dict is not None:
                    for var in yearly_data_path_dict.keys():
                        if var in datasets_to_include:
                            yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]
                            data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                            data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                            variable_dict[var] = list(data_arr)

                # reading static data and storing it in a dictionary
                if static_data_path_dict is not None:
                    for var in static_data_path_dict.keys():
                        if var in datasets_to_include:
                            static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                            data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                            data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                            variable_dict[var] = list(data_arr)

                predictor_df = pd.DataFrame(variable_dict)
                predictor_df = predictor_df.dropna()

                # saving input predictor csv
                monthly_output_csv = os.path.join(output_dir, f'predictors_{year}_{month}.csv')
                predictor_df.to_csv(monthly_output_csv, index=False)

    else:
        pass


def scale_monthy_peff_with_wateryr_peff_model(years_list, unscaled_peff_monthly_dir, unscaled_peff_water_yr_dir,
                                              scaled_peff_water_yr_dir, output_dir,
                                              skip_processing=False):
    """
    Scale effective precipitation (peff) monthly data using the water year peff fraction model.
    The model ensures imposing water year precipitation > water year peff

    :param years_list: List of years_list to process data for.
    :param unscaled_peff_monthly_dir: Filepath of original monthly peff (from monthly model) estimates' directory.
    :param unscaled_peff_water_yr_dir: Filepath of original water year peff, summed from monthly model, estimates'
                                       directory.
    :param scaled_peff_water_yr_dir: Filepath of updated water year peff, estimated from water year model, estimates'
                                     directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this process. Default set to False.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        months = range(1, 12 + 1)

        for yr in years_list:
            for mn in months:

                if (yr == 1985) and (mn <= 9):
                    continue

                elif (yr == 2020) and (mn >= 10):
                    continue

                else:
                    print(f'Scaling monthly Peff with water year Peff fraction model for year {yr}, month {mn}...')

                    # selecting the water year of total peff data based on month
                    if mn in range(10, 12 + 1):
                        peff_unbound_wy = glob(os.path.join(unscaled_peff_water_yr_dir, f'*{yr + 1}*.tif'))[0]
                        peff_unbound_wy_arr = read_raster_arr_object(peff_unbound_wy, get_file=False)

                        peff_bound_wy = glob(os.path.join(scaled_peff_water_yr_dir, f'*{yr + 1}*.tif'))[0]
                        peff_bound_wy_arr = read_raster_arr_object(peff_bound_wy, get_file=False)

                    elif mn in range(1, 9 + 1):
                        peff_unbound_wy = glob(os.path.join(unscaled_peff_water_yr_dir, f'*{yr}*.tif'))[0]
                        peff_unbound_wy_arr = read_raster_arr_object(peff_unbound_wy, get_file=False)

                        peff_bound_wy = glob(os.path.join(scaled_peff_water_yr_dir, f'*{yr}*.tif'))[0]
                        peff_bound_wy_arr = read_raster_arr_object(peff_bound_wy, get_file=False)

                    # selecting the monthly peff data
                    unscaled_peff_monthly = glob(os.path.join(unscaled_peff_monthly_dir, f'*{yr}_{mn}*.tif'))[0]
                    unscaled_peff_monthly_arr, raster_file = read_raster_arr_object(unscaled_peff_monthly)

                    # scaling monthly peff with bounded peff total
                    # the bounded peff total comes from water year model, where we impose water year peff < water year precip
                    scaled_peff_monthly_arr = np.where(~np.isnan(unscaled_peff_monthly_arr),
                                                       unscaled_peff_monthly_arr * peff_bound_wy_arr / peff_unbound_wy_arr,
                                                       -9999)
                    scaled_peff_monthly_arr[unscaled_peff_monthly_arr == -9999] = -9999

                    output_raster = os.path.join(output_dir, f'effective_precip_{yr}_{mn}.tif')
                    write_array_to_raster(scaled_peff_monthly_arr, raster_file, raster_file.transform, output_raster)


def clip_netGW_Irr_frac_for_basin(years, basin_shp, netGW_input_dir, basin_netGW_output_dir,
                                  resolution=model_res):
    """
    Clip netGW and irrigated fraction datasets for a basin, Clipping irrigation fraction data is optional.

    :param years: List of years_list to process data.
    :param basin_shp: Filepath of basin shapefile.
    :param netGW_input_dir: Directory path of Western US netGW estimates.
    :param basin_netGW_output_dir: Output directory path to save the clipped netGW estimates for the basin.
    :param resolution: model resolution.

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
                                       crs='EPSG:26912', output_datatype=gdal.GDT_Float32,
                                       use_ref_width_height=False)


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
    extract_dict = {'year': [], 'netGW_mm': [], 'pumping_mm': [], 'pumping_AF': []}

    # lopping through each year and storing data in a list
    for year in years:
        netGW_data = glob(os.path.join(basin_netGW_dir, f'*{year}*.tif'))[0]
        netGW_arr = read_raster_arr_object(netGW_data, get_file=False).flatten()

        year_list = [year] * len(netGW_arr)

        extract_dict['year'].extend(year_list)
        extract_dict['netGW_mm'].extend(list(netGW_arr))

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
        df = df[['year', 'netGW_mm']]

    # converting netGW mm to AF
    area_mm2_single_pixel = (2000 * 1000) * (2000 * 1000)  # unit in mm2
    df['netGW_AF'] = df['netGW_mm'] * area_mm2_single_pixel * 0.000000000000810714  # 1 mm3 = 0.000000000000810714 AF

    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    return output_csv


def aggregate_netGW_insitu_pumping_to_annualCSV_AZ(pixel_netGW_csv, annual_pumping_csv, basin_code,
                                                    area_basin_mm2, output_annual_csv):
    """
    Aggregate (by sum) pixel-wise annual netGW and in-situ pumping records for a basin
    to a annual csv.     *** used for HQR INA and Doug. AMA in AZ  ***

    *** provides annual netGW/in-situ pumping/usgs pumping (in AF/year and mm/year)
    and mean netGW/in-situ pumping/usgs pumping (in mm/year).

    :param pixel_netGW_csv: Filepath of csv holding pixel-wise annual netGW data for a basin.
    :param annual_pumping_csv: In-situ annually summed pumping csv (for all basins in AZ).
    :param basin_code: Either 'hqr' or 'doug' to select from 'HARQUAHALA INA' or 'DOUGLAS AMA'.
    :param area_basin_mm2: Area of the basin in mm2.
    :param output_annual_csv: Filepath of output annual total/mean netGW/pumping csv.

    :return: None.
    """
    print('Aggregating netGW and in-situ pumping to a annual csv...')

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

    # merging netGW estimates + in-situ pumping together
    yearly_df = netGW_df.merge(pump_df, on='year')

    # calculating m3 values
    yearly_df['netGW_m3'] = yearly_df['netGW_AF'] * 1233.48
    yearly_df['pumping_m3'] = yearly_df['pumping_AF'] * 1233.48

    # calculating mean netGW + mean pumping + mean USGS pumping (in mm)
    yearly_df['mean netGW_mm'] = yearly_df['netGW_AF'] * 1233481837547.5 / area_basin_mm2
    yearly_df['mean pumping_mm'] = yearly_df['pumping_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm

    # saving final csv
    yearly_df.to_csv(output_annual_csv, index=False)



def run_annual_csv_processing_AZ(years, basin_code, basin_shp,
                                 AZ_netGW_dir, annual_pumping_csv,
                                 main_output_dir, pixelwise_output_csv,
                                 final_annual_csv,
                                 skip_processing=False):
    """
    Run processes to compile a basins' netGW, pumping, and USGS pumping data at annual scale in a csv for
    Harquahala INA and Douglas AMA in Arizona.

    :param years: List of years_list to process data.
    :param basin_code: Basin keyword to get area and save processed datasets. Must be one of the following-
                        ['hqr', 'doug']
    :param basin_shp: Filepath of basin shapefile.
    :param AZ_netGW_dir: AZ netGW directory.
    :param annual_pumping_csv: Filepath of annual basin aggregated pumping database (csv).
    :param main_output_dir: Filepath of main output directory to store processed data for a basin.
    :param pixelwise_output_csv: Filepath of csv holding pixel-wise annual netGW and pumping data for a basin.
    :param final_annual_csv: Filepath of final output csv with annual netGW and in-situ pumping.
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
        makedirs([basin_netGW_dir])

        # # # # #  STEP 1 # # # # #
        # # Clip growing season netGW for the basin
        print('# # # # #  STEP 1 # # # # #')

        clip_netGW_Irr_frac_for_basin(years=years, basin_shp=basin_shp,
                                      netGW_input_dir=AZ_netGW_dir,
                                      basin_netGW_output_dir=basin_netGW_dir,
                                      resolution=model_res)

        # # # # #  STEP 2 # # # # #
        # # Compile pixel-wise growing season netGW and annual pumping in dataframes
        print('# # # # #  STEP 2 # # # # #')

        compile_pixelwise_basin_df_for_netGW_pumping(years=years, basin_netGW_dir=basin_netGW_dir,
                                                     basin_pumping_mm_dir=None,
                                                     basin_pumping_AF_dir=None,
                                                     output_csv=pixelwise_output_csv)

        # # # # #  STEP 3 # # # # #
        # # Compile the basin's pixelwise netGW and in-situ pumping to a common csv
        print('# # # # #  STEP 5 # # # # #')

        aggregate_netGW_insitu_pumping_to_annualCSV_AZ(pixel_netGW_csv=pixelwise_output_csv,
                                                       annual_pumping_csv=annual_pumping_csv,
                                                       basin_code=basin_code,
                                                       area_basin_mm2=basin_area_dict[basin_code],
                                                       output_annual_csv=final_annual_csv)
    else:
        pass


def make_line_plot_v1(y1, y2, year, fontsize, xlabel, ylabel, line_label_1, line_label_2,
                      figsize=(10, 4), y_lim=None, legend_pos='upper left',legend='on',
                      savepath=None, no_xticks=False, suptitle=None):

    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(year, y1, label=line_label_1, color='tab:blue', marker='^', linewidth=1)
    ax.plot(year, y2, label=line_label_2, color='tab:green', marker='^', linewidth=1)
    ax.set_xticks(year)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(y_lim)

    # legend
    if legend == 'on':
        ax.legend(loc=legend_pos, fontsize=(fontsize - 2))
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove bounding box
    sns.despine(offset=10, trim=True)  # turning of bounding box around the plots

    # suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    # xticks
    ax.set_xticklabels(labels=year, rotation=45, fontsize=fontsize)
    if no_xticks:
        ax.set_xticklabels(labels=[])

    plt.subplots_adjust(bottom=0.35)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True)
