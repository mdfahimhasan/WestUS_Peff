#!/bin/bash

## running the eff_precip_model.py script on CPU nodes

#SBATCH --partition=smi_all
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python m03_peff_adjust.py
