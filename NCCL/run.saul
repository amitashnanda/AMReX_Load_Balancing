#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 00:15:00 
#SBATCH -J FBtest
#SBATCH -o FBtest.o%A
##SBATCH -A m1759_g
#SBATCH -N 256 
#SBATCH -c 32
#SBATCH -q early_science 
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1

# Must also be on at compile-time
export CRAY_ACCEL_TARGET=nvidia80
# Only needed at run-time
export MPICH_GPU_SUPPORT_ENABLED=1

# Set maximum level use GPU Direct RDMA for Perlmutter
export NCCL_NET_GDR_LEVEL=PHB

EXE=./main3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=inputs

srun ${EXE} ${INPUTS}

# Inline in an salloc:
#srun -n 4 --gpu-bind=map_gpu:0,1,2,3 ./main3d.gnu.TPROF.MPI.CUDA.ex inputs
