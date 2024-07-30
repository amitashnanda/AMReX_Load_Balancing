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

    int ncomp;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
    }

    // This test specifically needs 2 ghost cells.
    nghost = {2,2,2};

//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************
    {
        Box domain(IntVect{0}, (d_size-=1));
        BoxArray ba(domain);
        ba.maxSize(mgs);

        int nranks = ParallelDescriptor::NProcs();

        long nboxes = ba.size();
        Vector<int> dst_map_a(nboxes, 0);
        Vector<int> dst_map_b(nboxes, 0);
        Vector<Real> weights_a(nboxes, 0);
        Vector<Real> weights_b(nboxes, 0);
        Vector<Real> local_wgtA, local_wgtB;

        int creator = ParallelDescriptor::IOProcessorNumber();

        if (ParallelDescriptor::IOProcessor()) {
            for (int i=0; i<nboxes; ++i)
            {
                dst_map_a[i] = amrex::Random_int(nranks);
                weights_a[i] = amrex::RandomNormal(5.0, 10.0);

                dst_map_b[i] = amrex::Random_int(nranks);
                weights_b[i] = amrex::RandomNormal(2.0, 0.5);
            }
        }

        ParallelDescriptor::Bcast(dst_map_a.data(), dst_map_a.size(), creator);
        ParallelDescriptor::Bcast(dst_map_b.data(), dst_map_b.size(), creator);
        ParallelDescriptor::Bcast(weights_a.data(), weights_a.size(), creator);
        ParallelDescriptor::Bcast(weights_b.data(), weights_b.size(), creator);

        DistributionMapping dm_a(dst_map_a);
        DistributionMapping dm_b(dst_map_b);

        MultiFab mf_a, mf_b;
        mf_a.define(ba, dm_a, ncomp, nghost);
        mf_b.define(ba, dm_b, ncomp, nghost);

        Periodicity period(piv);
        LayoutData<Real> ld_a(ba, dm_a);
        LayoutData<Real> ld_b(ba, dm_b);

        local_wgtA.resize(mf_a.local_size());
        for (MFIter mfi(mf_a); mfi.isValid(); ++mfi) {
            int idx = mfi.tileIndex();
            local_wgtA[idx] = amrex::RandomPoisson(2.3);
            ld_a[mfi] = local_wgtA[idx];
        }

        local_wgtB.resize(mf_b.local_size());
        for (MFIter mfi(mf_b); mfi.isValid(); ++mfi) {
            int idx = mfi.tileIndex();
            local_wgtB[idx] = amrex::RandomPoisson(-1.4);
            ld_b[mfi] = local_wgtB[idx];
        }

        // ======================================================

        amrex::Graph test_graph;

        test_graph.addFab(mf_a, "A-Before", weights_a, "A", ParallelDescriptor::MyProc());
        test_graph.addFab(mf_b, "B-Before", weights_b, "B", ParallelDescriptor::MyProc()+1);

        // FB 1 on A, FB 2 on B, PC between.
        test_graph.addFillBoundary("FB-Before, 1 Ghost", "A-Before", 1.0 + amrex::RandomNormal(0.1, 0.001),
                                   mf_a, IntVect{1,1,1}, period);
        test_graph.addFillBoundary("FB-Before, 2 Ghost", "B-Before", 2.0 + amrex::RandomNormal(0.1, 0.001),
                                   mf_b, IntVect{2,2,2}, period);
        test_graph.addParallelCopy("PC-Before", "B-Before", "A-Before", 3.0 + amrex::RandomNormal(0.1, 0.001),
                                   mf_b, mf_a);

        // print both ways
        test_graph.print("readable.graph");
        test_graph.print_table("table");

        // ======================================================
        // Balance both ways.e

        DistributionMapping dm_a_knap, dm_b_knap, dm_a_SFC, dm_b_SFC;

        Real currEff_a=0, currEff_b=0;
        Real newEff_a_knap=0, newEff_b_knap=0;
        Real newEff_a_SFC=0, newEff_b_SFC=0;

        dm_a_knap = dm_a.makeKnapSack(ld_a, currEff_a, newEff_a_knap);
        dm_b_knap = dm_b.makeKnapSack(ld_b, currEff_b, newEff_b_knap);
        dm_a_SFC = dm_a.makeSFC(ld_a, currEff_a, newEff_a_SFC);
        dm_b_SFC = dm_b.makeSFC(ld_b, currEff_b, newEff_b_SFC);

        // Define the diff

        // Build new fabs
        MultiFab mf_a_sfc, mf_a_knap, mf_b_sfc, mf_b_knap;
        mf_a_sfc.define(ba, dm_a_SFC, ncomp, nghost);
        mf_a_knap.define(ba, dm_a_knap, ncomp, nghost);
        mf_b_sfc.define(ba, dm_b_SFC, ncomp, nghost);
        mf_b_knap.define(ba, dm_b_knap, ncomp, nghost);

        test_graph.addFab(mf_a_sfc, "A-SFC", weights_a, "SFC on A", ParallelDescriptor::MyProc()-1);
        test_graph.addFab(mf_b_sfc, "B-SFC", weights_b, "SFC on B", ParallelDescriptor::MyProc()-2);
        test_graph.addFab(mf_a_knap, "A-Knap", weights_a, "Knap on A", ParallelDescriptor::MyProc()+3);
        test_graph.addFab(mf_b_knap, "B-Knap", weights_b, "Knap on B", ParallelDescriptor::MyProc()+5);

        test_graph.addFillBoundary("FB-SFC, 1 Ghost", "A-SFC", 0, mf_a_sfc, IntVect{1,1,1}, period);
        test_graph.addFillBoundary("FB-SFC, 2 Ghost", "B-SFC", 0, mf_b_sfc, IntVect{2,2,2}, period);
        test_graph.addFillBoundary("FB-Knap, 1 Ghost", "A-Knap", 1, mf_a_knap, IntVect{1,1,1}, period);
        test_graph.addFillBoundary("FB-Knap, 2 Ghost", "B-Knap", 1, mf_b_knap, IntVect{2,2,2}, period);

        // ======================================================

        // print both ways
        test_graph.print("readable.graph.2");
        test_graph.print_table("table_2");
    }
}
