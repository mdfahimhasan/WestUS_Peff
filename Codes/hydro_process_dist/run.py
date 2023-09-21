from Codes.hydro_process_dist.sw_dist import compile_USGS_WaterUse_data

# # Reference raster
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'

# # Compiling USGS county-scale GW data
skip_compiling_USGS_wateruse_data = True  # # Set to False to compile USGS water use data
years = [2005, 2010, 2015]

WaterUse_usgs_csv = compile_USGS_WaterUse_data\
    (yearlist=years, usgs_data_dir='../../Data_main/USGS_water_use_data',
     county_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_county_projected.shp',
     irrigated_data_dir='../../Data_main/Raster_data/Irrigated_agriculture/WestUS_Processed_irrigated_lands',
     county_id_data='../../Data_main/Compiled_data/reference_rasters/Western_US_countyID.tif',
     output_dir='../../Data_main/USGS_water_use_data',
     skip_compiling=skip_compiling_USGS_wateruse_data, ref_raster=WestUS_raster)
