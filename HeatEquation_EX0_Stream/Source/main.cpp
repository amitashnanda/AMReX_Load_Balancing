#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

#include "myfunc.H"

using namespace amrex;

/* Copy of amrex::Copy (from FabArray.H) to expose the cudaStream */
/* For testing, just always use fusing strategy. */
void HEqn_Copy (MultiFab& dst, MultiFab const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
{
    BL_PROFILE("HeatEquation::Copy()");

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion() && dst.isFusingCandidate()) {

        auto const& srcarr = src.const_arrays();
        auto const& dstarr = dst.arrays();

        Gpu::bg_stream().gpuSubmit( [=, &dst] (gpuStream_t stream) {
            ParallelFor(dst, nghost, numcomp,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
            {
                dstarr[box_no](i,j,k,dstcomp+n) = srcarr[box_no](i,j,k,srccomp+n);
            }, stream);
        });
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        // For this test, remove TilingIfNotGPU(). Need a way to add both?
        for (MFIter mfi(dst,MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.growntilebox(nghost);
            if (bx.ok())
            {
                auto const& srcFab = src.const_array(mfi);
                auto const& dstFab = dst.array(mfi);

                Gpu::bg_stream().gpuSubmit( [=] (gpuStream_t stream) {
                    ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept{   
                        dstFab(i,j,k,dstcomp+n) = srcFab(i,j,k,srccomp+n);
                    }, stream);
                });
            }
        }
    }
}


int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    // **********************************
    // SIMULATION PARAMETERS

    // number of cells on each side of the domain
    IntVect n_cell;

    // size of each box (or grid)
    IntVect max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;

    // Number of time steps to launch together.
    int launch_size;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size", max_grid_size);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // time step
        pp.get("dt",dt);

        // launch size
        launch_size = 1;
        pp.query("launch_size", launch_size);
    }

    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [0,1] in each direction.
    RealBox real_box({AMREX_D_DECL( 0., 0., 0.)},
                     {AMREX_D_DECL( 1., 1., 1.)});

    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi_old(ba, dm, Ncomp, Nghost);
    MultiFab phi_new(ba, dm, Ncomp, Nghost);

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA

    // loop over boxes
    for (MFIter mfi(phi_old); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& phiOld = phi_old.array(mfi);

        // set phi = 1 + e^(-(r-0.5)^2)
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real x = (i+0.5) * dx[0];
            Real y = (j+0.5) * dx[1];
#if (AMREX_SPACEDIM == 2)
            Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5))/0.01;
#elif (AMREX_SPACEDIM == 3)
            Real z= (k+0.5) * dx[2];
            Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))/0.01;
#endif
            phiOld(i,j,k) = 1. + std::exp(-rsquared);
        });
    }

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi_old, {"phi"}, geom, time, 0);
    }

    // **********************************
    // TIME LOOP 

    const double time_zero = amrex::second();

    for (int step = 1; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        phi_old.FillBoundary_stream(geom.periodicity());
/*
        // new_phi = old_phi + dt * Laplacian(old_phi)
        // Use MF Parallel Fors

        auto const& oldarr = phi_old.const_arrays();
        auto const& newarr = phi_new.arrays();

        Gpu::bg_stream().gpuSubmit( [=, &phi_new] (gpuStream_t stream) {
            ParallelFor(phi_new, IntVect{Nghost, Nghost, Nghost}, Ncomp,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
            {
                newarr[box_no](i,j,k) = oldarr[box_no](i,j,k,n) + dt *
                   ( (oldarr[box_no](i+1,j,k,n) - 2.*oldarr[box_no](i,j,k,n) + oldarr[box_no](i-1,j,k,n)) / (dx[0]*dx[0])
                   + (oldarr[box_no](i,j+1,k,n) - 2.*oldarr[box_no](i,j,k,n) + oldarr[box_no](i,j-1,k,n)) / (dx[1]*dx[1])
#if (AMREX_SPACEDIM == 3)
                   + (oldarr[box_no](i,j,k+1,n) - 2.*oldarr[box_no](i,j,k,n) + oldarr[box_no](i,j,k-1,n)) / (dx[2]*dx[2])
#endif
                   );
            }, stream);
        });
*/
        // loop over boxes
        for ( MFIter mfi(phi_old, MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& phiOld = phi_old.array(mfi);
            const Array4<Real>& phiNew = phi_new.array(mfi);

            Gpu::bg_stream().gpuSubmit( [=] (gpuStream_t stream) {
                // advance the data by dt
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    phiNew(i,j,k) = phiOld(i,j,k) + dt *
                        ( (phiOld(i+1,j,k) - 2.*phiOld(i,j,k) + phiOld(i-1,j,k)) / (dx[0]*dx[0])
                         +(phiOld(i,j+1,k) - 2.*phiOld(i,j,k) + phiOld(i,j-1,k)) / (dx[1]*dx[1])
#if (AMREX_SPACEDIM == 3)
                         +(phiOld(i,j,k+1) - 2.*phiOld(i,j,k) + phiOld(i,j,k-1)) / (dx[2]*dx[2])
#endif
                        );
                }, stream);
            });
        }

        // update time
        time = time + dt;

        // copy new solution into old solution
        HEqn_Copy(phi_old, phi_new, 0, 0, 1, IntVect{0, 0, 0});

//        amrex::Sleep(0.0025);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << " submitted at " << amrex::second() - time_zero << std::endl;

        Gpu::bg_stream().cpuSubmit( [=] () {
            amrex::Print() << "   Step " << step << " completed at " << amrex::second() - time_zero << std::endl;
        });

        if (step%launch_size == 0) {
            BL_PROFILE( "Waiting" );
            Gpu::bg_stream().sync();
        }

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, step);
        }
    }
}
