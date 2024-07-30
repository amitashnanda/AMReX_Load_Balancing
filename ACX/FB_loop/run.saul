#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 00:02:00
#SBATCH -J FBtest
#SBATCH -o FBtest.o%A-%a
#SBATCH -A nstaff_g
#SBATCH -N 2
#SBATCH -c 32
#SBATCH -q debug
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
##SBATCH --array=1-5

export MPICH_OFI_NIC_POLICY=NUMA

#EXE=./main3d.gnu.DEBUG.TPROF.MTMPI.CUDA.ex
EXE=./main3d.gnu.TPROF.MTMPI.CUDA.ex
INPUTS="inputs_2"
#ARENA="amrex.the_arena_is_managed=0"
#GPU_AWARE_MPI="amrex.the_arena_is_managed=0 amrex.use_gpu_aware_mpi=1"

#COMPUTE_SANITIZER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/compute-sanitizer/compute-sanitizer

export MPIACX_NFLAGS=256

#srun ${EXE} ${INPUTS} ${ARENA} ${GPU_AWARE_MPI}

#srun ${COMPUTE_SANITIZER} ${EXE} ${INPUTS}

#GPU_AWARE_MPI="amrex.the_arena_is_managed=0 amrex.use_gpu_aware_mpi=1"
#srun ${COMPUTE_SANITIZER} ${EXE} ${INPUTS} ${GPU_AWARE_MPI}

srun nsys profile --stats=true -t nvtx,cuda ${EXE} ${INPUTS}
