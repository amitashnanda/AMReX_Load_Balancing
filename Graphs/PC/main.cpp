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

    int src_ncomp, dst_ncomp, ncomp;
    double scaling = 0.0;
    std::string name = "fb";
    IntVect src_size, src_mgs, src_nghost;
    IntVect dst_size, dst_mgs, dst_nghost;
    IntVect piv;
    {
        ParmParse pp;
        pp.get("src_domain", src_size);
        pp.get("src_max_grid_size", src_mgs);
        pp.get("src_ncomp", src_ncomp);
        pp.get("src_nghost", src_nghost);

        pp.get("dst_domain", dst_size);
        pp.get("dst_max_grid_size", dst_mgs);
        pp.get("dst_ncomp", dst_ncomp);
        pp.get("dst_nghost", dst_nghost);

        pp.get("periodicity", piv);

        ncomp = amrex::min(src_ncomp, dst_ncomp);

        pp.query("name", name);
        pp.query("scaling", scaling);
        pp.query("ncomp", ncomp);
    }

//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************

    MultiFab src, dst;
    Periodicity period(piv);
    {
        Box src_domain(IntVect{0}, (src_size-=1));
        BoxArray src_ba(src_domain);
        src_ba.maxSize(src_mgs);

        Box dst_domain(IntVect{0}, (dst_size-=1));
        BoxArray dst_ba(dst_domain);
        dst_ba.maxSize(dst_mgs);
      
        DistributionMapping src_dm(src_ba);
        DistributionMapping dst_dm(dst_ba);

        src.define(src_ba, src_dm, src_ncomp, src_nghost);
        dst.define(dst_ba, dst_dm, dst_ncomp, dst_nghost);
    }

    amrex::Graph graph;
    graph.addParallelCopy(name, "src", "dst", scaling,
                          dst, src, src_ncomp, dst_ncomp, ncomp,
                          src_nghost, dst_nghost, period);

    graph.print_table(name);
}
