#include <AMReX_INT.H>
#include <AMReX_REAL.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Vector.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#include <Knapsack.H>

#include <queue>
#include <vector>

#if 0

void
LeastUsedTeams (amrex::Vector<int>                & rteam,
                amrex::Vector<amrex::Vector<int> >& rworker,
                int                                 nteams,
                int                                 nworkers)
{
#ifdef BL_USE_MPI
    BL_PROFILE("DistributionMapping::LeastUsedTeams()");

    AMREX_ALWAYS_ASSERT(ParallelContext::CommunicatorSub() == ParallelDescriptor::Communicator());

    Vector<Long> bytes(ParallelContext::NProcsSub());
    Long thisbyte = amrex::TotalBytesAllocatedInFabs()/1024;
    ParallelAllGather::AllGather(thisbyte, bytes.dataPtr(), ParallelContext::CommunicatorSub());

    std::vector<LIpair> LIpairV;
    std::vector<LIpair> LIworker;

    LIpairV.reserve(nteams);
    LIworker.resize(nworkers);

    rteam.resize(nteams);
    rworker.resize(nteams);

    for (int i(0); i < nteams; ++i)
    {
        rworker[i].resize(nworkers);

        Long teambytes = 0;
        int offset = i*nworkers;
        for (int j = 0; j < nworkers; ++j)
        {
            int globalrank = offset+j;
            Long b = bytes[globalrank];
            teambytes += b;
            LIworker[j] = LIpair(b,j);
        }

        Sort(LIworker, false);

        for (int j = 0; j < nworkers; ++j)
        {
            rworker[i][j] = LIworker[j].second;
        }

        LIpairV.push_back(LIpair(teambytes,i));
    }

    bytes.clear();

    Sort(LIpairV, false);

    for (int i(0); i < nteams; ++i)
    {
        rteam[i] = LIpairV[i].second;
    }
#else
    rteam.clear();
    rteam.push_back(0);
    rworker.clear();
    rworker.push_back(Vector<int>(1,0));
    amrex::ignore_unused(nteams,nworkers);
#endif
}

void
LeastUsedCPUs (int                 nprocs,
               amrex::Vector<int>& result)
{
    result.resize(nprocs);

#ifdef BL_USE_MPI
    BL_PROFILE("DistributionMapping::LeastUsedCPUs()");

    AMREX_ASSERT(nprocs <= amrex::ParallelContext::NProcsSub());

    amrex::Vector<amrex::Long> bytes(amrex::ParallelContext::NProcsSub());
    amrex::Long thisbyte = amrex::TotalBytesAllocatedInFabs()/1024;
    ParallelAllGather::AllGather(thisbyte, bytes.dataPtr(), ParallelContext::CommunicatorSub());

    std::vector<LIpair> LIpairV;

    LIpairV.reserve(nprocs);

    for (int i(0); i < nprocs; ++i)
    {
        LIpairV.push_back(LIpair(bytes[i],i));
    }

    bytes.clear();

    Sort(LIpairV, false);

    for (int i(0); i < nprocs; ++i)
    {
        result[i] = LIpairV[i].second;
    }

    if (flag_verbose_mapper) {
        Print() << "LeastUsedCPUs:" << std::endl;
        for (const auto &p : LIpairV) {
            Print() << "  Rank " << p.second << " contains " << p.first << std::endl;
        }
    }
#else
    for (int i(0); i < nprocs; ++i)
    {
        result[i] = i;
    }
#endif
}
#endif

void
LeastUsedCPUs (int                      nprocs,
               std::vector<amrex::Long> bytes,
               amrex::Vector<int>&      result,
               bool                     flag_verbose_mapper)
{
    result.resize(nprocs);

    if (nprocs > 1)
    {
        BL_PROFILE("DistributionMapping::LeastUsedCPUs()");

        AMREX_ASSERT(nprocs == bytes.size());

        std::vector<LIpair> LIpairV;

        LIpairV.reserve(nprocs);

        for (int i(0); i < nprocs; ++i)
        {
            LIpairV.push_back(LIpair(bytes[i],i));
        }

        bytes.clear();

        Sort(LIpairV, false);

        for (int i(0); i < nprocs; ++i)
        {
            result[i] = LIpairV[i].second;
        }

        if (flag_verbose_mapper) {
            amrex::Print() << "LeastUsedCPUs:" << std::endl;
            for (const auto &p : LIpairV) {
                amrex::Print() << "  Rank " << p.second << " contains " << p.first << std::endl;
            }
        }
    } else {
        result[0] = 0;
    }
}

