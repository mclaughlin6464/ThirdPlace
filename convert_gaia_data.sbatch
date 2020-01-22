#!/bin/bash
#SBATCH --job-name=convert_gaia_data
#SBATCH -p kipac,iric
#SBATCH --time=48:00:00
#SBATCH -o convert_gaia_data.out 
####SBATCH --exclusive
#SBATCH --cpus-per-task=8
####SBATCH --mem-per-cpu=MaxMemPerCPU

module load python/2.7.13
module load py-scipystack
module load hdf5/1.10.0p1

python /home/users/swmclau2/Git/ThirdPlace/convert_gaia_data.py
