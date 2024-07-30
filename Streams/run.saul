#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 00:02:00
#SBATCH -J FBtest
#SBATCH -o FBtest.o%A-%a
#SBATCH -A nstaff_g
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH -q debug
##SBATCH -q regular
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
##SBATCH --gpu-bind=none
##SBATCH --array=1-5

export CRAY_ACCEL_TARGET=nvidia80
export AMREX_CUDA_ARCH=8.0
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=1

EXE=./main3d.gnu.CUDA.ex
#INPUTS="inputs_mukul_2"
#GPU_AWARE_MPI=
GPU_AWARE_MPI="amrex.the_arena_is_managed=0 amrex.use_gpu_aware_mpi=1"

echo "GPU_AWARE_MPI = ${GPU_AWARE_MPI}"

srun --cpu-bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" 

#COMPUTE_SANITIZER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/compute-sanitizer/compute-sanitizer
#srun ${COMPUTE_SANITIZER} ${EXE} ${INPUTS} ${GPU_AWARE_MPI}

#srun nsys profile --stats=true -t nvtx,cuda ${EXE} ${INPUTS}
