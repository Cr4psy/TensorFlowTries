#!/bin/bash -l
#SBATCH -J CNN

# Defined the time allocation you use
#SBATCH -A <myCAC>

# 10 minute wall-clock time will be given to this job
#SBATCH -t 10:00

# Request K80 GPU accelerator
#SBATCH --gres=gpu:K80:2

# set tasks per node to 24 to disable hyperthreading
#SBATCH --ntasks-per-node=24

# load intel compiler and mpi

module add anaconda/py35/4.2.0
module add cudnn/5.1-cuda-8.0

conda create --name tf python=3.5
source activate tf
conda install matplotlib scipy pillow tensorflow-gpu

# Run program
mpirun -n 48 ~/Private/TensorFlowTries/KAGGLE/cnn_gpu.py
