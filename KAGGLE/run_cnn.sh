#!/bin/bash -l
#SBATCH -J CNN

# Defined the time allocation you use
#SBATCH -A edu17.DD2424

# 10 minute wall-clock time will be given to this job
#SBATCH -t 24:00:00

# set tasks per node to 24 to disable hyperthreading
#SBATCH --ntasks-per-node=24

# load intel compiler and mpi
module add cudnn/5.1-cuda-8.0
module load anaconda/py35/4.2.0
source activate tensorflow1.1

python ./cnn_gpu.py

source deactivate