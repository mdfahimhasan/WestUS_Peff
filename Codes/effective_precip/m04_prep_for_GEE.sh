#!/bin/bash

## running the m04_prep_for_GEE.py script on CPU nodes

#SBATCH --partition=smi_all
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python m04_prep_for_GEE.py
