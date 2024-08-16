// #include <AMReX.H>
// #include <AMReX_Random.H>
// #include <AMReX_MultiFab.H>
// #include <AMReX_ParmParse.H>
// #include <AMReX_ParallelDescriptor.H>
// #include <random>
// #include <string>
// #include <cstring>
// #include <omp.h>

// #include <Util.H>
// #include <Knapsack.H>
// #include <SFC.H>
// #include <BruteForce.H>


// #if defined(AMREX_USE_MPI) || defined(AMREX_USE_GPU)
// #error This is a serial test only.
// #endif

// using namespace amrex;

// void main_main();

// int main(int argc, char* argv[]) {
//     amrex::Initialize(argc, argv);
//     main_main();
//     amrex::Finalize();
// }

// void main_main() {
//     BL_PROFILE("main");

//     int ncomp;
//     double scaling = 0.0;
//     std::string name = "fb";
//     int nbins;
//     IntVect d_size, mgs, nghost, piv;
//     {
//         ParmParse pp;
//         pp.get("domain", d_size);
//         pp.get("max_grid_size", mgs);
        
//         pp.get("ncomp", ncomp);
//         pp.get("nghost", nghost);
//         pp.get("periodicity", piv);
//         pp.get("nbins", nbins);

//         pp.query("name", name);
//         pp.query("scaling", scaling);
//     }

//     Box domain(IntVect{0}, (d_size -= 1));
//     BoxArray ba(domain);
//     ba.maxSize(mgs);

//        int nitems = ba.size();
//     amrex::Print() << "Number of buckets: " << nbins << std::endl;
//     amrex::Print() << "Number of boxes: " << nitems << std::endl;

//     // BUILD WEIGHT DISTRIBUTION AND SORTING BYTES VECTOR
//     std::vector<amrex::Real> wgts(nitems);
//     std::vector<Long> bytes;

//     Real mean = 100000;
//     Real stdev = 4523;
//     for (int i = 0; i < nitems; ++i) {
//         wgts[i] = amrex::RandomNormal(mean, stdev);
//     }

//     // Scale weights and convert to Long for algorithms.
//     std::vector<Long> scaled_wgts = scale_wgts(wgts);

//     // No node_size in serial mode
//     std::vector<int> s_dmap = SFC_KnapSackProcessorMapDoIt(ba, scaled_wgts, nbins, &s_eff, true, false, bytes);

//     // Print the final distribution map
//     amrex::Print() << "Final distribution map:" << std::endl;
//     for (size_t i = 0; i < s_dmap.size(); ++i) {
//         amrex::Print() << "Box " << i << " -> Rank " << s_dmap[i] << std::endl;
//     }
// }

// // Helper function to scale weights
// std::vector<Long> scale_wgts(const std::vector<amrex::Real>& wgts) {
//     std::vector<Long> scaled_wgts(wgts.size());
//     for (size_t i = 0; i < wgts.size(); ++i) {
//         scaled_wgts[i] = static_cast<Long>(wgts[i] * 1e6); // Scale factor for example
//     }
//     return scaled_wgts;
// }


#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <random>
#include <string>
#include <cstring>  
#include <omp.h>
#include <fstream>
#include <numeric>
#include <cassert>

#include <Util.H>
#include <Knapsack.H>
#include <SFC.H>
#include <SFC_knapsack.H>

#if defined(AMREX_USE_MPI) || defined(AMREX_USE_GPU)
#error This is a serial test only.
#endif

using namespace amrex;
void main_main();

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);

    main_main();

    amrex::Finalize();
}

void main_main() {
    BL_PROFILE("main");

    int ncomp;
    double scaling = 0.0;
    std::string name = "fb";
    int nbins, nnodes, ranks_per_node;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
        pp.get("nbins", nbins);
        pp.get("nnodes", nnodes); 
        pp.get("ranks_per_node", ranks_per_node);  
        pp.query("name", name);
        pp.query("scaling", scaling);
    }


    Box domain(IntVect{0}, (d_size -= 1));
    BoxArray ba(domain);
    ba.maxSize(mgs);

    int nitems = ba.size();
    amrex::Print() << "Number of nodes: " << nnodes << std::endl;
    amrex::Print() << "Number of boxes: " << nitems << std::endl;
    amrex::Print() << "Ranks per node: " << ranks_per_node << std::endl;

    std::vector<amrex::Real> wgts(nitems);
    std::vector<Long> bytes;

    Real mean = 100000;
    Real stdev = 4523;
    for (int i = 0; i < nitems; ++i) {
        wgts[i] = amrex::RandomNormal(mean, stdev);
    }
    std::vector<Long> scaled_wgts = scale_wgts(wgts);

    amrex::Real sfc_eff = 0.0, knapsack_eff = 0.0;

    std::vector<int> sfc_knapsack_dmap = SFCProcessorMapDoIt(ba, scaled_wgts, nnodes, ranks_per_node, &sfc_eff, &knapsack_eff, true, false, bytes);

    // Print SFC and Knapsack efficiencies
    amrex::Print() << "SFC Efficiency: " << sfc_eff << std::endl;
    amrex::Print() << "Knapsack Efficiency: " << knapsack_eff << std::endl;
}
