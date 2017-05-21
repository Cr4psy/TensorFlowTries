#!/bin/bash -l
#SBATCH -J CNN

# Defined the time allocation you use
#SBATCH -A edu17.DD2424

#SBATCH --ntasks-per-node=2

# 10 minute wall-clock time will be given to this job
#SBATCH --time 24:00:00



# load intel compiler and mpi
module add cudnn/5.1-cuda-8.0
module load anaconda/py35/4.2.0
source activate tensorflow1.1

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=300 --execute CNNKaggle.ipynb

source deactivate
