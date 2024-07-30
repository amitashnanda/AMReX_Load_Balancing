# AMREX_HOME defines the directory in which we will find all the AMReX code.
# If you set AMREX_HOME as an environment variable, this line will be ignored
AMREX_HOME ?= ../../amrex

#DEBUG        = TRUE
#USE_CCACHE   = TRUE
USE_OMP      = FALSE
TINY_PROFILE = TRUE
USE_MPI      = TRUE
USE_CUDA     = TRUE
BL_NO_FORT   = TRUE
COMP         = gnu
DIM          = 3

USE_NCCL     = FALSE
USE_NVSHMEM  = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

# =====================================================
# Extra Additions

ifdef USE_NCCL
  USE_NCCL := $(strip $(USE_NCCL))
else
  USE_NCCL := FALSE
endif

ifdef USE_NVSHMEM
  USE_NVSHMEM := $(strip $(USE_NVSHMEM))
else
  USE_NVSHMEM := FALSE
endif


ifeq ($(USE_NCCL),TRUE)
  EXTRACXXFLAGS += -I$(NCCL_DIR)/include
  LINKFLAGS += -L$(NCCL_DIR)/lib -lnccl
  DEFINES += -DAMREX_USE_NCCL
endif

ifeq ($(USE_NVSHMEM),TRUE)
#  NVSHMEM_DIR := $(CUDATOOLKIT_HOME)/../../comm_libs/nvshmem/
  NVSHMEM_DIR := /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/nvshmem
  EXTRACXXFLAGS += -I$(NVSHMEM_DIR)/include
  LINKFLAGS += -L$(NVSHMEM_DIR)/lib -lnvshmem -lnvidia-ml
  DEFINES += -DAMREX_USE_NVSHMEM
endif
