#!/bin/bash

# 10min walltime:
#SBATCH --time 10:00:00

# 1 node:
#SBATCH --nodes 1

# 1 slot for MPI ranks per node:
#SBATCH --ntasks-per-node 1

# 4 CPUs per MPI rank:
#SBATCH --cpus-per-task 4

# 1 GPU per node:
#SBATCH --gres gpu:1 

# 4GB of RAM per CPU:
#SBATCH --mem-per-cpu=4000

# select partition:
#SBATCH --partition=gpu

# jobname:
#SBATCH -J gpuN1C4G1-train-maps

echo "Show CPU ids visible to the job:"
numactl --show
echo "Note, that the requested 4 CPUs correspond to 4 virtual cores \
    (threads), and thus only 2 physical cores."

echo -e "\nallocated GPUs are exclusively visible to this job, other GPUs \
    are not visible:"
nvidia-smi

echo -e "\nload  modules to use CUDA, e.g.:"
module load gcc cuda
