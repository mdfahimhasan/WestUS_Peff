# A Data-driven Approach for Estimating Effective Precipitation in the Western United States and Applications in Irrigation Groundwater Management

## Abstract
Effective precipitation, defined as the portion of ET derived from precipitation, is a crucial component of the irrigation water balance and controls the amount of water required for irrigation. Quantification and validation of effective precipitation is challenging as it comprises of multiple fluxes – precipitation, surface runoff, and deep percolation. Existing empirical and process-based effective precipitation estimation approaches lack complexity, accuracy, and validation, especially for large-scale applications.  Moreover, there is no readily available high-resolution effective precipitation dataset for the United States (US), despite its importance in water management. In this study, we focus on the Western United States (Western US), a region severely impacted by excessive groundwater withdrawal, and developed a data-driven approach for estimating effective precipitation at high spatio-temporal resolution. The framework incorporates a machine learning model that extracts effective precipitation information from non-irrigated croplands and provides its estimates for the irrigated croplands of the Western US at a 2 km resolution and on a monthly scale from 2000 to 2020. The model performs satisfactorily, with a training R2 of 0.841 and a root mean squared error (RMSE) of 11.11,  and a test R2 of 0.831 and an RMSE of 11.23. Furthermore, we analyzed the principal drivers of effective precipitation to understand its dynamics in irrigated landscapes. To validate the effective precipitation estimates, we measured groundwater pumping over seven basins of the Western US using a water balance approach that incorporates the model-generated effective precipitation estimates. A comparison of our estimated pumping with in-situ records shows R2 of 0.87 and 0.75 in volumetric and depth spaces, respectively. This validates our effective precipitation dataset using in-situ groundwater pumping records. The successful implementation and satisfactory performance of our model and estimated pumping values show the potential of our effective precipitation estimation approach in irrigation water management. Furthermore, the high-resolution effective precipitation dataset will play a pivotal role in satellite-based groundwater pumping monitoring, thereby supplementing in-situ groundwater monitoring efforts.

__Keywords:__ Effective precipitation, Groundwater; Irrigation; Water use; Remote sensing; Machine learning.

## Publications
in progress

## Organizations
<img src="readme_figs/CSU-Signature-C-357-617.png" height="80"/> <img src="readme_figs/Official-DRI-Logo-for-Web.png" height="70"/>

## Funding
<img src="readme_figs/NASA-Logo-Large.png" height="80"/>

## Running the repository

### Repository structure
The repository has five main modules described as follows-

__1. utils -__ consists of scripts that helps in basic raster, vector, and statistical operation. It also holds the `ml_ops` scripts which has the machine learning functions.

__2. data_download_preprocess -__ consists of scripts that have functions to download datasets from GEE, including OpenET, and to further pre-process the datasets. The `run_download_preprocess.py` is the main driver script that has to be used to download and pre-process all required datasets.

__3. effective_precip -__ consists of functions that are required specifically for the effective precipitation model. The `eff_precip_model.py` is the main driver script of the 
effective precipitation model.

__4. sw_irrig -__ consists of functions that are required for dictributing USGS HUC12 level surface water irrigation data to 2 km pixel scale. The `SW_Irr.py` is the main driver file.

__5. netGW -__ consists of the `netGW_Irr.py` script that has the functions to estimate consumptive groundwter use for irrigation at 2 km resolution using a water balance appraoch.  

The __utils__ module do not need any execution. The latter modules are required to be executed using the respective driver files to unvail the full funtionality of the model. The repository has other auxiliary folders with scripts that are some data processing, result analysis,and plotting purposes.

### Dependencies
__operating system:__ All scripts are fully functional in windows and linux environments. In linux environment, gdal needs to be installed separately and the appropriate 'gdal_path' needs to be set in necessary scripts. For some functions, e.g. the `shapefile_to_raster()` in `utils > raster_ops.py` and associated scripts (`results_analysis > netGW_pumping_compile.py`), gdal system call has to enabled/installed specifically to run them in linux environment. Note that all modules, except `results_analysis`, have been implemented/checked using both windows and linux environment (using conda environment). However, the authors recommend exercising discretion when setting up the environment and run the scripts.

__conda environment:__ A _conda environment_ set up using [Anaconda](https://www.anaconda.com/products/individual) with python 3.9 has been used to implement this repositories. Required libraries needed to be installed to run this repository are - dask, dask-geopandas, earthengine-api, fastparquet, gdal, matplotlib, numpy, pandas, rasterio, scikit-learn, shapely, seaborn. 

Note that the `.ipynb` scripts will require installaion of jupyter lab within the conda environment.

### Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). You also have to create a gcloud project to use the GEE API. 

