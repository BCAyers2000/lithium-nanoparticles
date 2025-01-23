#!/bin/bash
#SBATCH --job-name=[100]
#SBATCH --mem 0             
#SBATCH --nodes=3
#SBATCH --cpus-per-task=1                       
#SBATCH --ntasks-per-node=128       
#SBATCH --exclusive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --account=e89-soto
#SBATCH --partition=standard                

module purge
module load craype-x86-rome
module load PrgEnv-gnu/8.3.3
module load cray-fftw/3.3.10.3
module load cray-hdf5-parallel/1.12.2.1

export OMP_STACKSIZE=128M
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export SLURM_CPU_FREQ_REQ=2250000
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

/work/e89/e89/ba3g18/Installations/miniforge3/envs/Ibrav-ase/bin/python Electrolyte.py


