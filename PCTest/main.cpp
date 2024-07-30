#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
void main_main ();

// ================================================

Real MFdiff(const MultiFab& lhs, const MultiFab& rhs,
            int strt_comp, int num_comp, int nghost, const std::string name = "")
{
    MultiFab temp(rhs.boxArray(), rhs.DistributionMap(), rhs.nComp(), nghost);
    temp.ParallelCopy(lhs);
    temp.minus(rhs, strt_comp, num_comp, nghost);

    if (name != "")
        { amrex::VisMF::Write(temp, std::string("pltfiles/" + name)); }

    Real max_diff = 0;
    for (int i=0; i<num_comp; ++i)
    {
        Real max_i = std::abs(temp.max(i));
        max_diff = (max_diff > max_i) ? max_diff : max_i;
    }

    return max_diff; 
}

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

    int ncell, ncomp;
    int nboxes = 0;
    int nghost = 0;
    int cellref = 256;
    Vector<int> maxcomp;
    {
        ParmParse pp;
        pp.get("ncell", ncell);
        pp.query("nboxes", nboxes);
        pp.getarr("maxcomp", maxcomp);
        pp.query("cell_ref", cellref);
    }

    if (nboxes == 0)
        { nboxes = ParallelDescriptor::NProcs(); } 

    MultiFab mf_src, mf_dst;

// ***************************************************************
    // Build the Multifabs and Geometries.
    {
        ncomp = std::pow(cellref/ncell, 3);
        Box domain(IntVect{0}, IntVect{ncell-1, ncell-1, nboxes*(ncell-1)});
        BoxArray ba(domain);
        ba.maxSize(ncell);

        amrex::Print() << "domain = " << domain << std::endl;
        amrex::Print() << "boxsize = " << ncell << std::endl;
        amrex::Print() << "nranks = " << nboxes << std::endl;
        amrex::Print() << "ncomp = " << ncomp << std::endl;
        amrex::Print() << "cell_ref = " << cellref << std::endl << std::endl;

        DistributionMapping dm_src(ba);

        Vector<int> dst_map = dm_src.ProcessorMap();
        for (int& b : dst_map)
        {
           if (b != ParallelDescriptor::NProcs()-1) 
               { b++; } 
           else 
               { b=0; }
        }

        DistributionMapping dm_dst(dst_map);

        Real val = 13.0;
        mf_src.define(ba, dm_src, ncomp, nghost);
        mf_src.setVal(val++);

        mf_dst.define(ba, dm_dst, ncomp, nghost);
        mf_dst.setVal(val++);

        amrex::Print() << "dm = " << dm_src << std::endl;
        Vector<int> count(nboxes, 0);
        for (int& p: dst_map)
            { count[p]++; }
        for (int i=0; i<count.size(); ++i)
            { amrex::Print() << "count[" << i << "]: " << count[i] << std::endl; }
    }

    FabArrayBase::MaxComp = ncomp*2;

    {   
        BL_PROFILE("**** Test - 1st");
        mf_dst.ParallelCopy(mf_src);
    }

    {
        BL_PROFILE("**** Test - 2nd");
        mf_dst.ParallelCopy(mf_src);
    }
    {
        BL_PROFILE("**** Test - 3rd");
        mf_dst.ParallelCopy(mf_src);
    }

    for (int& fmc : maxcomp)
    {
        FabArrayBase::MaxComp = fmc;

        BL_PROFILE(std::string("^^^^ ParallelCopy - ") + std::to_string(fmc));
        mf_dst.ParallelCopy(mf_src);
    }

    amrex::Print() << "Error in old PC: " 
                   << MFdiff(mf_src, mf_dst, 0, ncomp, nghost) << std::endl;
}
