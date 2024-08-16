#ifndef AMREX_KNAPSACK
#define AMREX_KNAPSACK

#include <AMReX_INT.H>
#include <AMReX_Vector.H>

using LIpair = std::pair<amrex::Long, int>;

struct LIpairLT
{
    bool operator () (const LIpair& lhs,
                      const LIpair& rhs) const noexcept
    {
        return lhs.first < rhs.first;
    }
};

struct LIpairGT
{
    bool operator () (const LIpair& lhs,
                      const LIpair& rhs) const noexcept
    {
        return lhs.first > rhs.first;
    }
};

// =============================================================

void
Sort (std::vector<LIpair>& vec,
      bool                 reverse);

void
knapsack (const std::vector<amrex::Long>&  wgts,
          int                              nprocs,
          std::vector< std::vector<int> >& result,
          amrex::Real&                     efficiency,
          bool                             do_full_knapsack = true,
          int                              nmax = std::numeric_limits<int>::max(),
          const amrex::Real&               max_efficiency = 0.9);
// void
// bruteForce (const std::vector<amrex::Long>&  wgts,
//           int                              nprocs,
//           std::vector< std::vector<int> >& result,
//           amrex::Real&                     efficiency,
//           bool                             do_full_knapsack = true,
//           int                              nmax = std::numeric_limits<int>::max(),
//           const amrex::Real&               max_efficiency = 0.9);

std::vector<int>
KnapSackDoIt (const std::vector<amrex::Long>& wgts,
              int                             nprocs,
              amrex::Real&                    efficiency,
              bool                            do_full_knapsack = true,
              int                             nmax = std::numeric_limits<int>::max(),
              bool                            flag_verbose_mapper = false,
              bool                            sort = false,
              const std::vector<amrex::Long>& bytes = std::vector<amrex::Long>());
//std::vector<int>
// BruteForceDoIt (const std::vector<amrex::Long>& wgts,
//               int                             nprocs,
//               amrex::Real&                    efficiency,
//               bool                            do_full_knapsack = true,
//               int                             nmax = std::numeric_limits<int>::max(),
//               bool                            flag_verbose_mapper = false,
//               bool                            sort = false,
//               const std::vector<amrex::Long>& bytes = std::vector<amrex::Long>());

#endif