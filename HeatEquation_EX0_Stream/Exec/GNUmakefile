# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= /global/homes/k/kngott/amrex/amrex/

DEBUG        = FALSE
TINY_PROFILE = TRUE
USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = TRUE
COMP         = gnu
DIM          = 3

MPI_THREAD_MULTIPLE = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
