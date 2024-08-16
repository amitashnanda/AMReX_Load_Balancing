#ifndef AMREX_SFC_KNAPSACK
#define AMREX_SFC_KNAPSACK

#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_Morton.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelContext.H>

#include <SFC.H>
#include <Knapsack.H>
#include <LeastUsed.H>



std::vector<int>
SFCProcessorMapDoItCombined (const amrex::BoxArray&          boxes,
                     const std::vector<amrex::Long>& wgts,
                     int                             nnodes,
                     int                             ranks_per_node,
                     amrex::Real*                    sfc_eff,
                     amrex::Real*                    knapsack_eff,
                     bool                            flag_verbose_mapper,
                     bool                            sort,
                     const std::vector<amrex::Long>& bytes);


#endif