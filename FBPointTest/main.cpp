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

// ***************************************************************
    {
        Box domain(IntVect{0}, d_size-1);
        BoxArray ba(domain);
        ba.maxSize(mgs);

        Periodicity period(piv);
        DistributionMapping dm(ba);

        MultiFab mf;
        mf.define(ba, dm, ncomp, nghost);
        mf.setVal(3.14159);

        double start_time = amrex::second(); 

        amrex::Print() << "BoxArray = " << ba << std::endl;

        // ======================================================

        for (int i=0; i<steps; ++i)
        {
//            BL_PROFILE_REGION("FB #"+std::to_string(i));

            double time = amrex::second();

            mf.FillBoundary(period);
            amrex::ParallelDescriptor::Barrier();

            double end_time = amrex::second();

            amrex::Print() << "FB #" << i << " = " << double(end_time - time)
                           << "\tclock: " << double(end_time - start_time) << std::endl;
        }
    }

}
