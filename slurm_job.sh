#!/bin/bash

# 24hrs walltime:
#SBATCH --time 48:00:00

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
#SBATCH -J gpuN1C4G1-train-photorealism

# stdouts
# #SBATCH --output=out.log
# #SBATCH --error=err.log

# Notification
# Mail alert at BEGIN|END|FAIL|ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.mokarian-forooshani@hzdr.de

LOG_FILE=./job_${SLURM_JOB_NAME}.log

function Log {
	local level=$1
	local msg=$2
	echo $(date --rfc-3339=seconds):${level} ${msg} >> ${LOG_FILE}
}

Log INFO "JOB START"
Log INFO "JOB NAME = ${SLURM_JOB_NAME}"

echo "Show CPU ids visible to the job:"
numactl --show
echo "Note, that the requested 4 CPUs correspond to 4 virtual cores \
    (threads), and thus only 2 physical cores."

echo -e "\nallocated GPUs are exclusively visible to this job, other GPUs \
    are not visible:"
nvidia-smi

module load gcc python/3.11 cuda/11.7

echo -e "\nmodule loaded for batch job:"
module list

Log INFO "changing to directory: ${SLURM_SUBMIT_DIR}"
cd $SLURM_SUBMIT_DIR

source ./venv/bin/activate

python train.py >> ${LOG_FILE} 2>&1

Log INFO "JOB FINISH"