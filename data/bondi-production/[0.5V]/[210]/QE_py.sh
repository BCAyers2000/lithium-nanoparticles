#!/bin/bash -l

#$ -N miller-210			
#$ -cwd
#$ -P Gold
#$ -pe mpi 360	
#$ -S /bin/bash
#$ -l h_rt=30:00:00 		
#$ -A Soton_allocation

module purge
module load gerun
module load gcc-libs/10.2.0
module load compilers/intel/2022.2 
module load fftw/3.3.10-impi/intel-2022
module load mpi/intel/2019/update6/intel

export OMP_NUM_THREAD=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

/lustre/scratch/mmm1182/Installations/miniforge3/envs/Workflow/bin/python3.12 Electrolyte.py
