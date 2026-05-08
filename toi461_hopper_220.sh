#!/bin/bash
#
## How many slots needed - S is the number of slots desired
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --job-name=TOI461_hopper_220
## memory per cpu; total memory assigned=mem-per-cpu * S
##  Default is 2048M. Specify memory required if more than the default is needed
## Replace X with the amount you require, specify in Megabytes
#SBATCH --mem-per-cpu=6096M
#### NOTE:  $SCRATCH defaults to your directory on /scratch space
#SBATCH --output /scratch/mabdall/TOI461_hopper_220_%N-%j.output
#SBATCH --error  /scratch/mabdall/TOI461_hopper_220_%N-%j.error

module load python/3.8.6-ye
module load julia/1.8.0
source /home/USER-ID/VirtualEnvName/bin/activate

unset Display
echo Starting Script
julia toi461_hopper_220.jl
echo SCRIPT COMPLETE
