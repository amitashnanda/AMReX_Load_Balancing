#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 00:02:00
#SBATCH -J FBtest
#SBATCH -o FBtest.o%A-%a
#SBATCH -A nstaff_g
#SBATCH -N 2
#SBATCH -c 32
##SBATCH -q debug
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
##SBATCH --gpu-bind=none
#SBATCH --array=1-5
##SBATCH --reservation=NERSC_hackathon_2023_day1_gpu

export CRAY_ACCEL_TARGET=nvidia80
export AMREX_CUDA_ARCH=8.0
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1

EXE=./main3d.gnu.TPROF.MTMPI.CUDA.ex
#EXE=./main3d.gnu.DEBUG.TPROF.MTMPI.CUDA.ex

INPUTS="inputs"
#INPUTS="inputs_lots"

#GPU_AWARE_MPI=
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

echo -e "GPU_AWARE_MPI = ${GPU_AWARE_MPI}\n"

srun --cpu-bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" 

#COMPUTE_SANITIZER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/compute-sanitizer/compute-sanitizer
#srun ${COMPUTE_SANITIZER} ${EXE} ${INPUTS} ${GPU_AWARE_MPI}

##  For -t mpi:
##  export LD_LIBRARY_PATH=$PE_PERFTOOLS_MPICH_LIBDIR:$PE_MPICH_GTL_DIR_nvidia80:$LD_LIBRARY_PATH
##
##  --stats=true -> instructs Nsight to print a summary of the application's performance to STDOUT

#srun --cpu-bind=cores nsys profile -t nvtx,cuda --mpi-impl=mpich bash -c "
#        export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
#        ${EXE} ${INPUTS} ${GPU_AWARE_MPI}"
