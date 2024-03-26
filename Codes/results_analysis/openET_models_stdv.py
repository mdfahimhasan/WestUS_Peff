import os
import numpy as np
from glob import glob

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster


def stdv_of_openET_models(years, model_dir_dict, netGW_dir, stdv_output_dir):
    for year in years:
        print(f'estimating stdv and mean for openET models for {year}...')
        makedirs([stdv_output_dir])

        # collecting each model's data for a year
        ssebop_data = glob(os.path.join(model_dir_dict['SSEBOP'], f'*{year}*'))[0]
        eemetric_data = glob(os.path.join(model_dir_dict['EEMETRIC'], f'*{year}*'))[0]
        geesebal_data = glob(os.path.join(model_dir_dict['GEESEBAL'], f'*{year}*'))[0]
        ptjpl_data = glob(os.path.join(model_dir_dict['PTJPL'], f'*{year}*'))[0]
        sims_data = glob(os.path.join(model_dir_dict['SIMS'], f'*{year}*'))[0]
        disalexi_data = glob(os.path.join(model_dir_dict['DISALEXI'], f'*{year}*'))[0]

        # reading each model's data
        ssebop_arr = read_raster_arr_object(ssebop_data, get_file=False)
        eemetric_arr = read_raster_arr_object(eemetric_data, get_file=False)
        geesebal_arr = read_raster_arr_object(geesebal_data, get_file=False)
        ptjpl_arr = read_raster_arr_object(ptjpl_data, get_file=False)
        sims_arr = read_raster_arr_object(sims_data, get_file=False)
        disalexi_arr = read_raster_arr_object(disalexi_data, get_file=False)

        # estimating standard deviation of the models at annual scale
        stacked_arrays = np.stack([ssebop_arr, eemetric_arr, geesebal_arr,
                                   ptjpl_arr, sims_arr, disalexi_arr], axis=0)
        stdv_per_element = np.std(stacked_arrays, axis=0)

        # estimating annual mean
        mean_per_element = np.mean(stacked_arrays, axis=0)

        # reading netGW data. will be used as filter
        netGW_data = glob(os.path.join(netGW_dir, f'*{year}*'))[0]
        netGW_arr, file = read_raster_arr_object(netGW_data)

        # forcing nan values (-9999) with netGW array. In netGW array, nan values has been replaced with zero.
        stdv_per_element[netGW_arr == 0] = -9999
        mean_per_element[netGW_arr == 0] = -9999

        # saving stdv and mean for each year
        stdv_raster = os.path.join(stdv_output_dir, f'openET_stdv_{year}.tif')
        write_array_to_raster(raster_arr=stdv_per_element, raster_file=file, transform=file.transform,
                              output_path=stdv_raster)

        mean_raster = os.path.join(stdv_output_dir, f'openET_mean_{year}.tif')
        write_array_to_raster(raster_arr=mean_per_element, raster_file=file, transform=file.transform,
                              output_path=mean_raster)


if __name__ == '__main__':
    openet_model_dict = {'SSEBOP': '../../Data_main/Raster_data/SSEBOP/WestUS_annual',
                         'EEMETRIC': '../../Data_main/Raster_data/EEMETRIC/WestUS_annual',
                         'GEESEBAL': '../../Data_main/Raster_data/GEESEBAL/WestUS_annual',
                         'PTJPL': '../../Data_main/Raster_data/PTJPL/WestUS_annual',
                         'SIMS': '../../Data_main/Raster_data/SIMS/WestUS_annual',
                         'DISALEXI': '../../Data_main/Raster_data/DISALEXI/WestUS_annual'}

    years_list = [2016, 2017, 2018, 2019, 2020]
    netGW_dir = '../../Data_main/Raster_data/NetGW_irrigation/WesternUS'
    output_dir = '../../Data_main/Raster_data/OpenET_stdv_mean'

    stdv_of_openET_models(years=years_list, model_dir_dict=openet_model_dict,
                          netGW_dir=netGW_dir, stdv_output_dir=output_dir)
