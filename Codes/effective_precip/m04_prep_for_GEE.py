from m00_eff_precip_utils import process_monthly_peff_rasters_to_multiband_forGEE

"""
This script converts the effective precipitation monthly datasets into multi-band raster for each model years_list 
from 2000 to 2020. The multi-band rasters are uploaded to GEE as assets and later converted into 
a single multi-band image for all years_list. For 2020, data is available up to September as the monthly datasets
were scaled using a water year fraction model which ends in water year 2020 (doesn't simulate water year 2021).

Details on downloading the dataset with GEE is available on - https://github.com/mdfahimhasan/WesternUS_NetGW
"""

years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
         2022, 2023, 2024]

model_version = 'v19'  ######
peff_monthly_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly_scaled'
output_dir = f'../../Data_main/Raster_data/Effective_precip_prediction_WestUS/{model_version}_monthly_scaled_multibands'

if __name__ == '__main__':
    process_monthly_peff_rasters_to_multiband_forGEE(years, peff_monthly_dir, output_dir, nodata=-9999)