#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
void main_main ();

// ================================================

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}

// ================================================

void main_main ()
{

    BL_PROFILE("main");

    int ncomp, mgs, nwork=20;
    IntVect lo, hi, bvect;
    {
        ParmParse pp;
        pp.get("lo", lo);
        pp.get("hi", hi);
        pp.get("ncomp", ncomp);
        pp.get("max_grid_size", mgs);
        pp.get("boundary", bvect);
        pp.query("nwork", nwork);
    }

    MultiFab mf;
    iMultiFab bmask;
    Box domain(lo, hi);

    BoxArray ba(domain);
    ba.maxSize(mgs);
    DistributionMapping dm(ba);

    Real start_val = 3.14159;
    mf.define(ba, dm, ncomp, 0);
    mf.setVal(start_val);

    bmask.define(ba, dm, 1, 0);
    bmask.setVal(0);

    amrex::Print() << std::endl;
    amrex::Print() << "domain = " << domain << std::endl;
    amrex::Print() << "ncomp = " << ncomp << std::endl;
    amrex::Print() << "ba = " << ba << std::endl;
    amrex::Print() << "mgs = " << mgs << std::endl;

    // Build Mask
    {
        BL_PROFILE("** Mask Build");

        for (MFIter mfi(mf); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            auto mfv = bmask.array(mfi);

            Box ibx = bx;
            for (int i=0; i<AMREX_SPACEDIM; ++i)
            {
                ibx.growLo(i, -bvect[i]);
                ibx.growHi(i, -bvect[i]);
            }

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!ibx.contains(i,j,k))
                    { mfv(i,j,k) = 1; }
            }); 
        }
    }

    long cells = ba.numPts();
    long bcells = bmask.sum(0)*ncomp; 

    amrex::Print() << std::endl;
    amrex::Print() << "Boundary Cells: " << bcells << std::endl; 
    amrex::Print() << "Total Cells: " << cells << std::endl;
    amrex::Print() << "Boundary %: " << 100*(double(bcells)/double(cells)) << std::endl;

    amrex::Print() << std::endl;
    amrex::Print() << "nwork: " << nwork << std::endl;
    amrex::Print() << " ========================= " << std::endl;

    double t_all, t_bound, t_int;

    for (int nw=1; nw<=nwork; ++nw)
    {
        amrex::Print() << "nwork = " << nw << std::endl;
    
        // Time w/o mask
        {
            t_all = amrex::second();
            BL_PROFILE("** Full");
    
            for (MFIter mfi(mf); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.validbox();
                auto arr = mf.array(mfi); 
    
                amrex::ParallelFor(bx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
                {
                    Real y = arr(i,j,k,c);
                    Real x = 1.0;
                    for (int n = 0; n < nw; ++n) {
                        Real dx = -(x*x-y) / (2.*x);
                        x += dx;
                    }
                    arr(i,j,k,c) = x;
                });        
            }
            t_all = amrex::second() - t_all;
        }
    
        mf.setVal(start_val);
    
        // Time of interior
        {
            t_int = amrex::second();
            BL_PROFILE("** Interior");
    
            for (MFIter mfi(mf); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.validbox();
                auto arr = mf.array(mfi); 
                auto mask = bmask.const_array(mfi);
    
                amrex::ParallelFor(bx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
                {
                    if (!mask(i,j,k)) {
                        Real y = arr(i,j,k,c);
                        Real x = 1.0;
                        for (int n = 0; n < nw; ++n) {
                            Real dx = -(x*x-y) / (2.*x);
                            x += dx;
                        }
                        arr(i,j,k,c) = x;
                    }
                });        
            }
            t_int = amrex::second() - t_int;
        }
    
        mf.setVal(start_val);
    
        // Time of boundary
        {
            t_bound = amrex::second();
            BL_PROFILE("** Boundary");
    
            for (MFIter mfi(mf); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.validbox();
                auto arr = mf.array(mfi); 
                auto mask = bmask.const_array(mfi);
    
                amrex::ParallelFor(bx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
                {
                    if (mask(i,j,k)) {
                        Real y = arr(i,j,k,c);
                        Real x = 1.0;
                        for (int n = 0; n < nw; ++n) {
                            Real dx = -(x*x-y) / (2.*x);
                            x += dx;
                        }
                        arr(i,j,k,c) = x;
                    }
                });        
            }
            t_bound = amrex::second() - t_bound;
        }
    
        amrex::Print() << "All: " << t_all << std::endl;
        amrex::Print() << "Interior: " << t_int << std::endl;
        amrex::Print() << "Boundary: " << t_bound << std::endl;
        amrex::Print() << "% Interior: " << 100*(t_int/t_all) << std::endl;
        amrex::Print() << "% Boundary: " << 100*(t_bound/t_all) << std::endl;
        amrex::Print() << std::endl;
    }

}
