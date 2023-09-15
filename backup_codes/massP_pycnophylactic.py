import numpy as np
from scipy import ndimage

from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

no_data_value = -9999
model_res = 0.02000000000000000389  # in deg, 2 km
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def run_mass_preserve(countyID_raster, wateruse_obsv_raster, crop_raster, developed_raster, output_MassPreserved_file,
                      ref_raster=WestUS_raster):
    print('|| MASS-PRESERVING AREAL WEIGHTING')

    # Reading necessary files
    countyID_arr = read_raster_arr_object(countyID_raster, get_file=False)
    WaterUse_arr = read_raster_arr_object(wateruse_obsv_raster, get_file=False)
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    # Integrating land use data for mass preserving
    crop_arr = read_raster_arr_object(crop_raster, get_file=False)
    developed_arr = read_raster_arr_object(developed_raster, get_file=False)

    landUse_arr = np.where((crop_arr > 0) | (developed_arr > 0), 1, ref_arr)

    unique, counts = np.unique(countyID_arr[~np.isnan(countyID_arr) & (landUse_arr == 1)], return_counts=True)
    counts = dict(zip(unique, counts))

    countsmp = np.copy(countyID_arr)

    for polid in counts.keys():
        countsmp[countsmp == polid] = counts[polid]  # creates an array with number of total pixels in that county

    masspdataset = WaterUse_arr / countsmp
    masspdataset[(landUse_arr != 1) & ~np.isnan(landUse_arr)] = 0

    write_array_to_raster(masspdataset, ref_file, ref_file.transform, output_MassPreserved_file)

    return output_MassPreserved_file


def polygonValuesByID(wateruse_arr, countyID_arr):
    uniqueids = np.unique(countyID_arr[~np.isnan(countyID_arr)])

    county_wateruse_disagg_dict = {}
    for polid in uniqueids:
        county_wateruse_disagg_dict[polid] = wateruse_arr[countyID_arr == polid][1]

    return county_wateruse_disagg_dict


def statsByID(wateruse_arr, countyID_arr, stat='sum'):
    unique, counts = np.unique(np.unique(countyID_arr[~np.isnan(countyID_arr)]), return_counts=True)
    counts = dict(zip(unique, counts))

    county_stats = {}
    for polid in counts.keys():
        if stat == 'sum':
            county_stats[polid] = np.nansum(wateruse_arr[countyID_arr == polid])
        else:
            print('Invalid statistic')

    return county_stats


def run_pycnophylactic_interp(massP_raster, wateruse_obsv_raster, countyID_raster, final_pycno_raster,
                              ref_raster=WestUS_raster):
    print('| PYCNOPHYLACTIC INTERPOLATION')
    countyID_arr = read_raster_arr_object(countyID_raster, get_file=False)
    WaterUse_arr = read_raster_arr_object(wateruse_obsv_raster, get_file=False)
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    pycno_arr = read_raster_arr_object(massP_raster, get_file=False)
    oldpycno_arr = pycno_arr

    county_wateruse_before_pycno_dict = polygonValuesByID(WaterUse_arr, countyID_arr)  # summed value in a county
    pycnomask = np.copy(countyID_arr)
    pycnomask[~np.isnan(pycnomask)] = 1

    niter = 10
    converge = 0.01

    for it in range(1, niter + 1):
        print('| - Iteration', it)

        # Calculate the mean of the cells in the 3 by 3 neighborhood
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        # mask = np.expand_dims(mask, axis=2)

        pycno_arr = ndimage.generic_filter(pycno_arr, np.nanmean, footprint=mask, mode='constant', cval=np.NaN)

        # Summarizes the values within each polygon (county)
        county_stats = statsByID(pycno_arr, countyID_arr, 'sum')  # summed value in a county

        # Divide the previous (before pycnophylactic applied) summed county value by new summed county value
        # Divide the true polygon values by the estimated polygon values (= ratio)
        polygonratios = {k: county_wateruse_before_pycno_dict[k] / county_stats[k] for k in county_stats.keys() &
                         county_wateruse_before_pycno_dict}

        # Multiply ratio by the different cells within each polygon
        for polid in polygonratios:
            pycno_arr[countyID_arr == polid] = (pycno_arr[countyID_arr == polid] * polygonratios[polid])

        pycno_arr = pycno_arr * pycnomask

        # Check if the algorithm has converged
        error_mae = np.nanmean(abs(pycno_arr - oldpycno_arr))  # mean absolute error
        rangeds = np.nanmax(oldpycno_arr) - np.nanmin(oldpycno_arr)
        stopcrit = converge  # * rangeds
        print('Error:', error_mae)

        if ((it > 1) and (error_mae < stopcrit)):
            break
        else:
            oldpycno_arr = pycno_arr

    pycno_arr = np.where(np.isnan(pycno_arr) & (ref_arr == 0), ref_arr, pycno_arr)

    write_array_to_raster(pycno_arr, raster_file=ref_file, transform=ref_file.transform, output_path=final_pycno_raster)


