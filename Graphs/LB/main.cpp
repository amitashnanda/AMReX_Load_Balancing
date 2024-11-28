


#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <random>
#include <string>
#include <cstring>
#include <time.h>  
#include <omp.h>
#include <fstream>
#include <numeric>
#include <cassert>

#include <Util.H>
#include <Knapsack.H>
#include <SFC.H>
#include <SFC_knapsack.H>
#include<painterPartition.H>

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
    Real mean, stdev;
    int nruns = 1;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
        // pp.get("nbins", nbins);
        pp.get("nnodes", nnodes); 
        pp.get("mean", mean);
        pp.get("stdev",stdev);
        pp.get("ranks_per_node", ranks_per_node);  
        pp.query("nruns",nruns);
        pp.query("name", name);
        pp.query("scaling", scaling);
    }
    amrex::Print() << "Mean: " << mean << std::endl;
    amrex::Print() << "Stdev: " << stdev << std::endl;
    amrex::Print() << "No of Run: " << nruns << std::endl;


    // srand(time(NULL));
    amrex::ResetRandomSeed(rand());

    // amrex::ResetRandomSeed(27182182459045);

    int nmax = std::numeric_limits<int>::max();
    Real k_eff = 0.0;
    Real s_eff = 0.0;

    Box domain(IntVect{0}, (d_size -= 1));
    BoxArray ba(domain);
    ba.maxSize(mgs);

    int nitems = ba.size();
    amrex::Print() << "Number of nodes: " << nnodes << std::endl;
    amrex::Print() << "Number of boxes: " << nitems << std::endl;
    amrex::Print() << "Ranks per node: " << ranks_per_node << std::endl;

    std::vector<amrex::Real> wgts(nitems);
    std::vector<Long> bytes;

    // Real mean = 100000;
    // Real stdev = 4523; // average case

    // // Real stdev = 25231; //for the worst case 

    // // Real stdev = 250; //for the best case

for (int r = 0; r<nruns; r++) {

    amrex::Print() << "\n=== Starting Run " << r + 1 << " ===\n";

    amrex::ResetRandomSeed(rand());


    for (int i = 0; i < nitems; ++i) {
        wgts[i] = amrex::RandomNormal(mean, stdev);
        amrex::Print()<<wgts[i]<<" , ";
    }
    std::vector<Long> scaled_wgts = scale_wgts(wgts);
    amrex::Print()<<" Scaled Weights: ";
      for (int i=0; i<nitems; ++i) {
        
        amrex::Print()<<scaled_wgts[i]<<" , ";
       
    }

    amrex::Real sfc_eff = 0.0, knapsack_eff = 0.0;
    int node_size = 0;
    double time_start=0;

    time_start = amrex::second();
    std::vector<int> k_dmap = KnapSackDoIt(scaled_wgts, nnodes, k_eff, true, nmax, true, false, bytes);
    amrex::Print()<<" Final Knapsack time: " << amrex::second() - time_start << std::endl;

    time_start = amrex::second();
    std::vector<int> s_dmap = SFCProcessorMapDoIt(ba, scaled_wgts, nnodes, &s_eff, node_size, true, false, bytes);
    amrex::Print()<<" Final SFC time: " << amrex::second() - time_start << std::endl;

    time_start = amrex::second();
    std::vector<int> vec=painterPartition(ba,scaled_wgts,nnodes);
    amrex::Print()<<" Final SFC+Painter time: " << amrex::second() - time_start << std::endl;

    time_start = amrex::second();
    std::vector<int> sfc_knapsack_dmap = SFCProcessorMapDoItCombined(ba, scaled_wgts, nnodes, ranks_per_node, &sfc_eff, &knapsack_eff, true, false, bytes);
    amrex::Print()<<" Final SFC+Knapsack_Combined time: " << amrex::second() - time_start << std::endl;

    amrex::Print() << "\n=== End of Run " << r + 1 << " ===\n";
    amrex::Print() << "======================================\n\n";

}

    // Print SFC and Knapsack efficiencies
    // amrex::Print() << "SFC Efficiency: " << sfc_eff << std::endl;
    // amrex::Print() << "Knapsack Efficiency: " << knapsack_eff << std::endl;
}
