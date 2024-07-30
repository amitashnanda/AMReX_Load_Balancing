#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Graph.H>

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

    bool cross = false;
    int ncomp, cross_in = 0;
    double scaling = 0.0;
    std::string name = "fb";
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);

        pp.query("name", name);
        pp.query("scaling", scaling);
        pp.query("cross", cross_in);
    }

    cross = bool(cross_in);

// ***************************************************************
  
    Box domain(IntVect{0}, (d_size-=1));
    BoxArray ba(domain);
    ba.maxSize(mgs);

    DistributionMapping dm(ba);
    Periodicity period(piv);

    FabArrayBase fab(ba, dm, ncomp, nghost);

//    MultiFab mf(ba, dm, ncomp, nghost);

    amrex::Graph graph;
    graph.addFillBoundary("FB", "mf", scaling, sizeof(Real), sizeof(Real),
                           fab, 0, ncomp, nghost, period, cross);

    graph.print_table(name);
}
