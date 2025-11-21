# Physics-constrained effective precipitation model for the Western United States

## Abstract
Effective precipitation, defined as the portion of evapotranspiration (ET) derived from precipitation, is an important part of the agricultural water balance and affects the amount of water required for irrigation. Due to hydrologic complexity, effective precipitation is challenging to quantify and validate using existing empirical and process-based methods. Moreover, there is no readily available high-resolution effective precipitation dataset for the United States (US), despite its importance for determining requirements and consumptive use of irrigation water. In this study, we developed a framework that incorporates multiple hydrologic states and fluxes within a two-step machine learning approach that accurately predicts effective precipitation for irrigated croplands of the Western US at ~2 km spatial resolution and monthly time steps from 2000 to 2024. Additionally, we analyzed the factors influencing effective precipitation to understand its dynamics in irrigated landscapes. To further assess effective precipitation estimates, we estimated groundwater pumping for irrigation in seven basins of the Western US with a water balance model that incorporates model-generated effective precipitation. A comparison of our estimated pumping volumes with in-situ records indicates good skill, with R2 of 0.78 and PBIAS of –15%. Though challenges remain in predicting and assessing effective precipitation, the satisfactory performance of our model and good skill in estimated pumping illustrate the application and potential of integrating satellite data and machine learning with a physically-based water balance to estimate key water fluxes. The effective precipitation dataset developed in this study has the potential to be used with satellite-based actual ET data for estimating consumptive use of irrigation water at large temporal and spatial scales and enable best available science-informed water management decisions.

__Keywords:__ Effective precipitation, Groundwater; Irrigation; Water use; Remote sensing; Machine learning.

## Effective precipitation map
<img src="https://raw.githubusercontent.com/mdfahimhasan/WesternUS_NetGW/master/Codes/results_analysis/Peff_animation/Peff_monthly_animation.gif" height="500"/>
Figure: Machine learning model generated monthly effective precipitation estimates from 2010 to 2014 at 2 km spatial resolution.

## Citations
- Hasan, M. F., Smith, R. G., Majumdar, S., Huntington, J. L., Alves Meira Neto, A., & Minor, B. A. (2025). Satellite data and physics-constrained machine learning for estimating effective precipitation in the Western United States and application for monitoring groundwater irrigation. Agricultural Water Management, 319, 109821. https://doi.org/10.1016/j.agwat.2025.109821
- Majumdar, S., Smith, R.G., Hasan, M.F., Wogenstahl, C., & Conway, B.D. (2025). A long-term database of groundwater pumping, consumptive use, effective precipitation, and irrigation efficiencies in Arizona derived from remote sensing and machine learning. In prep. for Nature Scientific Data.

## Organizations
<img src="readme_figs/CSU-Signature-C-357-617.png" height="90"/> <img src="readme_figs/Official-DRI-Logo-for-Web.png" height="80"/>

## Funding
<img src="readme_figs/NASA-Logo-Large.png" height="80"/>

## Running the repository

### Repository structure
The repository has five main modules described as follows-

__1. utils -__ consists of scripts that helps in basic raster, vector, and statistical operation. It also holds the `ml_ops` scripts which has the machine learning functions.

__2. data_download_preprocess -__ consists of scripts that have functions to download datasets from GEE, including OpenET, and to further pre-process the datasets. The `run_download_preprocess.py` is the main driver script that has to be used to download and pre-process all required datasets.

__3. effective_precip -__ consists of functions that are required specifically for the effective precipitation model. The effective precipitation is estimated by a 3-step model. First, the `m01_peff_model_monthly.py` script estmates effective precipitation at monthly scale. The monthly estimates do not follow water balance (water year precipitation > water year effective precipitation) in some regions. So, at the second step,  the `m02_peff_frac_model_water_yr.py` script simulates a water year-scale effective precipitation fraction model. This water year-scale model is used to impose water balance over the monthly estimates using the `m03_peff_adjust.py` script. These three files have to be run in sequence to generate the monthly effective precipitation estimates.

__4. sw_irrig -__ consists of functions that are required for dictributing USGS HUC12 level surface water irrigation data to 2 km pixel scale. The `SW_Irr.py` is the main driver file.

__5. netGW -__ consists of the `netGW_Irr.py` script that has the functions to estimate consumptive groundwter use for irrigation at 2 km resolution using a water balance appraoch.  

The __utils__ module do not need any execution. The latter modules are required to be executed using the respective driver files to unvail the full funtionality of the model. The repository has other auxiliary folders with scripts that are used for some data processing, result analysis,and plotting purposes.

### Dependencies
__operating system:__ Most scripts are fully functional in windows and linux environments except some. In linux environment, gdal needs to be installed separately and the appropriate 'gdal_path' needs to be set in necessary scripts. For some functions, e.g. the `shapefile_to_raster()` in `utils > raster_ops.py` and associated scripts (`results_analysis > netGW_pumping_compile.py`), gdal system call has to enabled/installed specifically to run them in linux environment. Note that all scripts, except the scripts in `results_analysis` module, have been implemented/checked using both windows and linux environment (using conda environment). In addition, the ALE plot generation in `m01_peff_model_monthly.py` and `m02_peff_frac_model_water_yr.py` scripts do not respond (keep running indifinitely) in linux environment (probably due to scikit-explain versioning issue); therefore, set `skip_plot_ale = True` when running the monthly and water year models in linux environment.

The authors recommend exercising discretion when setting up the environment and run the scripts.

__conda environment:__ A _conda environment_, set up using [Anaconda](https://www.anaconda.com/products/individual) with python 3.9, has been used to implement this repositories. Required libraries needed to be installed to run this repository are - dask, dask-geopandas, earthengine-api, fastparquet, rasterio, gdal, shapely, geopandas, numpy, pandas, scikit-learn, lightgbm, scikit-explain, matplotlib, seaborn. 

Note that running the `.ipynb` scripts will require installation of jupyter lab within the conda environment.

### Google Earth Engine authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). You also have to create a gcloud project to use the GEE API. 

## Data availability
The monthly effective precipitation estimates for all months of 2000 to 2024 (up to September in 2024) are available to download through this [GEE script](https://code.earthengine.google.com/6352c6b7b5efb066e738de692df46a72). Non-GEE users can acccess the dataset from this [HydroShare repo](https://www.hydroshare.org/resource/c33ce80f5ae44fe6ab2e5dd3c128eb0b/). __Note that the operationally updated dataset will be maintained and updated exclusively in Google Earth Engine (GEE).__

