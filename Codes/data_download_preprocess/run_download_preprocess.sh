#!/bin/bash

## running the run_download_preprocess.py script on CPU nodes

#SBATCH --partition=smi_all
#SBATCH --ntasks=30
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python run_download_preprocess.py

