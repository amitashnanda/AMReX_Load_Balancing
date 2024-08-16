#ifndef AMREX_LEASTUSED
#define AMREX_LEASTUSED

#include <AMReX_INT.H>
#include <AMReX_Vector.H>

void
LeastUsedCPUs (int                      nprocs,
               std::vector<amrex::Long> bytes,
               amrex::Vector<int>&      result,
               bool                     flag_verbose_mapper = false);
#endif
