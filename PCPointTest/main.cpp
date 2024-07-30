#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

using namespace amrex;
void main_main ();

// ================================================

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}

void main_main ()
{

    BL_PROFILE("main");

    int ncomp, steps;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
        pp.get("steps", steps);
    }

//    amrex::ResetRandomSeed(27182182459045);
    int nranks = ParallelDescriptor::NProcs();

// ***************************************************************
    {
        Box domain(IntVect{0}, d_size-1);
        BoxArray ba(domain);
        ba.maxSize(mgs);

        Periodicity period(piv);
        DistributionMapping dm_src(ba);

        MultiFab mf_src;
        mf_src.define(ba, dm_src, ncomp, nghost);
        mf_src.setVal(3.14159);

        Vector<int> pmap = dm_src.ProcessorMap(); 
        for (auto& rank : pmap)
        {
            if (rank == nranks-1) {
                rank = 0;
            } else {
                rank++;
            }
        }

        MultiFab mf_dst;
        DistributionMapping dm_dst(pmap);
        mf_dst.define(ba, dm_dst, ncomp, nghost);
        mf_dst.setVal(2*3.14159);

        double start_time = amrex::second(); 

        // ======================================================

        for (int i=0; i<steps; ++i)
        {
//            BL_PROFILE_REGION("FB #"+std::to_string(i));

            double time = amrex::second();

            mf_dst.ParallelCopy(mf_src, period);
            amrex::ParallelDescriptor::Barrier();

            double end_time = amrex::second();

            amrex::Print() << "PC #" << i << " = " << double(end_time - time)
                           << "\tclock: " << double(end_time - start_time)
                           << "\top time: " << double(time - start_time)
                           << " : " << double(time - start_time) + double(0.005)
                           << std::endl;
        }
    }

}
