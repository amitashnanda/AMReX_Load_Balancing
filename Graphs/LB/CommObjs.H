#include <AMReX_Config.H>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Periodicity.H>
#include <AMReX_Print.H>
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>

//===============================
// FB

void
make_fb (amrex::FabArrayBase::FB& my_fb,
         const amrex::FabArrayBase& fa, const amrex::IntVect& nghost,
         const amrex::Periodicity& period, int nprocs, int MyProc,
         const amrex::Vector<int>& imap, bool cross,
         bool enforce_periodicity_only, bool override_sync,
         bool multi_ghost);

void
define_fb (amrex::FabArrayBase::FB& my_fb,
           const amrex::FabArrayBase& fa, int MyProc, amrex::Vector<int> imap);

void
define_epo (amrex::FabArrayBase::FB& my_fb,
            const amrex::FabArrayBase& fa, int MyProc, amrex::Vector<int> imap);

void
tag_one_box (amrex::FabArrayBase::FB & my_fb,
             int krcv, amrex::BoxArray const& ba, amrex::DistributionMapping const& dm,
             bool build_recv_tag, int MyProc);

void
define_os (amrex::FabArrayBase::FB & my_fb,
           const amrex::FabArrayBase& fa,
           int MyProc, amrex::Vector<int> imap, int nprocs);

//===============================
// CPC

void setup_cpc (amrex::FabArrayBase::CPC& cpc);

void
make_cpc (amrex::FabArrayBase::CPC& my_cpc,
          const amrex::BoxArray& ba_dst,      const amrex::DistributionMapping& dm_dst,
          const amrex::Vector<int>& imap_dst, const amrex::IntVect& dstng,
          const amrex::BoxArray& ba_src,      const amrex::DistributionMapping& dm_src,
          const amrex::Vector<int>& imap_src, const amrex::IntVect& srcng,
          const amrex::Periodicity& period,   bool to_ghost_cells_only, int MyProc);
