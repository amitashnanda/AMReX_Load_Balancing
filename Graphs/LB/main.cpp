#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <random>
#include<string>
#include <cstring>  // For std::strcpy
//#include <AMReX_Graph.H>
#include <omp.h>

#include <Util.H>
#include <Knapsack.H>
#include <SFC.H>
#include<BruteForce.H>
//#include <CommObjs.H>

#if defined(AMREX_USE_MPI) || defined(AMREX_USE_GPU)
#error This is a serial test only. 
#endif


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
    double scaling = 0.0;
    std::string name = "fb";
    int nbins;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
        pp.get("nbins",nbins);

        pp.query("name", name);
        pp.query("scaling", scaling);
    }


//    amrex::ResetRandomSeed(27182182459045);

// ***************************************************************

/*  BUILD COMM PATTERN AND STORE IN GRAPH
    NEED TO ALTER FOR SERIAL
    
    Box domain(IntVect{0}, (d_size-=1));
    BoxArray ba(domain);
    ba.maxSize(mgs);

    DistributionMapping dm(ba);
    Periodicity period(piv);

    MultiFab mf(ba, dm, ncomp, nghost);

    amrex::Graph graph;
    graph.addFillBoundary("FB", "mf", scaling,
                          mf);

    graph.print_table(name);
*/

// ***************************************************************
/* BUILD A BOXARRAY AND WEIGHTS TO LOAD BALANCE */

    // TEST INFO (PARMPARSE)
    //int nbins = 2;
    int nmax = std::numeric_limits<int>::max();
    Real k_eff = 0.0;
    Real s_eff = 0.0;
    Real target_eff = 0.9;

   
    // BUILD BOXARRAY FOR SFC & COMM PATTERNS
    //IntVect d_size(256, 256, 256);  // we need to have this as a variable 
                                    // or read from a file
    //IntVect mgs(128,128,128);
    //IntVect mgs(64,64,64);
    //IntVect mgs(32,32,32);

 //amrex::Print() << "max_grid_size " << mgs << std::endl;
    Box domain(IntVect{0}, (d_size-=1));
    BoxArray ba(domain);
    ba.maxSize(mgs);

    int nitems = ba.size();
    amrex::Print()<<"Number of buckets: "<<nbins<<std::endl;
    amrex::Print()<<"Number of boxes: "<<nitems<<std::endl;
    

    // BUILD WEIGHT DISTRIBUTION AND SORTING BYTES VECTOR
    std::vector<amrex::Real> wgts(nitems);
    //std::vector<amrex::double> guess(nitems);
    std::vector<Long> bytes;
    //std::vector<Long> guess(nitems);

    Real mean = 100000;  /// We need to have read from file
    Real stdev = 4523;
    for (int i=0; i<nitems; ++i) {
        wgts[i] = amrex::RandomNormal(mean, stdev);
       // guess[i] = amrex::RandomNormal(mean, stdev);
    }
    
    // Scale weights and convert to Long for algorithms.
    std::vector<Long> scaled_wgts = scale_wgts(wgts);

    //generate array of guesses with normal distribution
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean,stdev);
   // amrex::Print()<<"Guess values: ";
    // for (int i=0;i<nitems;i++){
    //     guess[i]=distribution(generator);  //guess value based on the normal distribution
	// if (guess[i]<0){
	//     amrex::Print()<<"negative guess, make mean higher or standard devation smaller"<<std::endl;
    //         exit(0);
	// } 
    // //amrex::Print()<<guess[i]<<" ";
    // }
    // amrex::Print()<<std::endl;
    // SFC parameter -- default = 0
    int node_size = 0;

//scaled_wgts: The scaled weights of the 8 boxes.
//nbins: The number of bins.
// k_eff: Variable to store the efficiency
// nmax: Maximum number of items per bin
// output is k_dmap: Distribution map indicating which bin each weight (box) is assigned to.
// k_eff: The efficiency of the distribution.


    std::vector<int> k_dmap = KnapSackDoIt(scaled_wgts, nbins, k_eff, true, nmax, true, false, bytes);
    std::vector<int> s_dmap = SFCProcessorMapDoIt(ba, scaled_wgts, nbins, &s_eff, node_size, true, false, bytes);
//    // std::vector<int> bruteForce_dmap = BruteForceDoIt(guess, nbins, k_eff, true, nmax, true, false, bytes);
    
    
//     BruteForceDoIt(nbins,nitems,mean,stdev,scaled_wgts.data());
    

   // get_all_combos(N,nr,guess,ranks,combo_fn);  // N=total number of boxes, 
                                                  // nr= number of buckets 
//                                                 //guess=based on normal distribution 
//                                                 //ranks= collection of ranks number suppose 0-63 for 64 ranks
#if 0
// ***************************************************************
/* PC comm pattern from CPCs */

    // Use DistributionMapping, or a stand-in?
    //      -- ownership/indexarrays DM arrays are inaccurate when serial. 
    // Build with an int vector?
    amrex::Vector<int> src_dist(nitems);
    amrex::Vector<int> dst_dist(nitems);

    for (int i=0; i<nitems; ++i) {
        src_dist[i] = amrex::Random_int(nbins);
        dst_dist[i] = amrex::Random_int(nbins);
    }
/*
    for (int i=0; i<nitems; ++i) {
        amrex::Print() << "S/D[" << i << "]: " << src_dist[i] << " " << dst_dist[i] << std::endl;
    }
*/
    DistributionMapping src_dm(src_dist);
    DistributionMapping dst_dm(dst_dist);

    amrex::Print() << "Source DM: " << src_dm << std::endl;
    amrex::Print() << "Dest DM: " << dst_dm << std::endl;

    FabArrayBase src(ba, src_dm, 1, {1,1,1});
    FabArrayBase dst(ba, dst_dm, 1, {1,1,1});

    Graph graph;
    graph.addFab(src, "src", sizeof(double));
    graph.addFab(dst, "dst", sizeof(double));

    IntVect dstng(0,0,0), srcng(0,0,0);
    bool to_ghost_cells_only = false;
    Periodicity period;

    // Build Comm Pattern -- Build CPC from all ranks.
    for (int MyProc=0; MyProc<nbins; ++MyProc) {

        FabArrayBase::CPC cpc_by_rank(amrex::BoxArray(), {1,1,1},
                                      amrex::DistributionMapping(),
                                      amrex::DistributionMapping());

        amrex::Vector<int> src_idx(0), dst_idx(0);
        for (int i=0; i<nitems; ++i) {
            // Create src & dst indexarrays for each rank.
            if (src_dm[i] == MyProc) { src_idx.push_back(i); }
            if (dst_dm[i] == MyProc) { dst_idx.push_back(i); }

        }

        make_cpc(cpc_by_rank,
                 ba, dst_dm, dst_idx, dstng,
                 ba, src_dm, src_idx, srcng,
                 period, to_ghost_cells_only, MyProc);

        if (MyProc == 0) {
            graph.addEdgeList("CopyTest", "src", "dst", 1.0, cpc_by_rank, 1);
        } else {
            graph.appendEdgeList("CopyTest", "src", "dst", 1.0, cpc_by_rank, 1);
        }

    }
    graph.print_table("CopyTest");
    graph.clear();

// ***************************************************************
/* FB comm patterns from FBs */


    graph.addFab(src, "src", sizeof(double));

    // Build Comm Pattern -- Build CPC from all ranks.
    for (int MyProc=0; MyProc<nbins; ++MyProc) {

        FabArrayBase::FB fb_by_rank(amrex::FabArrayBase(), {0,0,0},
                                    false, period, false, false, false);

        amrex::Vector<int> src_idx(0);
        for (int i=0; i<nitems; ++i) {
            // Create src & dst indexarrays for each rank.
            if (src_dm[i] == MyProc) { src_idx.push_back(i); }
        }

        IntVect ng(1,1,1);
        Periodicity period;

        bool cross = false;
        bool epo = false;
        bool os = false;
        bool multi_ghost = false;

        make_fb(fb_by_rank,
                src, ng, period, nbins, MyProc,
                src_idx, cross, epo, os, multi_ghost);


        if (MyProc == 0) {
            graph.addEdgeList("FBTest", "src", "src", 1.0, fb_by_rank, 1);
        } else {
            graph.appendEdgeList("FBTest", "src", "src", 1.0, fb_by_rank, 1);
        }
    }
    graph.print_table("FBTest");
#endif
// ***************************************************************

#if 0
    // Print individual data and collect bin data.
    Real wgt_avg = 0;
    Real real_eff = 0;
    std::vector<Real> bwgts(nbins); 
    for (int i=0; i<nitems; ++i) {
        bwgts[k_dmap[i]] += wgts[i]; 
        wgt_avg += wgts[i];
    }
    wgt_avg /= nbins;

    amrex::Print() << "BINS = (Avg: " << wgt_avg << ")" << std::endl;
    for (int i=0; i<nbins; ++i) {
        amrex::Print() << i << ": " << bwgts[i] << std::endl;
        real_eff = std::max(real_eff, bwgts[i]);
    }
    real_eff = double(1.0)-(real_eff - wgt_avg)/wgt_avg;
    amrex::Print() << "EFF = " << real_eff << std::endl;
#endif
// ***************************************************************

}




#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <random>
#include <string>
#include <cstring>
#include <omp.h>

#include <Util.H>
#include <Knapsack.H>
#include <SFC.H>
#include <BruteForce.H>


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
    int nbins;
    IntVect d_size, mgs, nghost, piv;
    {
        ParmParse pp;
        pp.get("domain", d_size);
        pp.get("max_grid_size", mgs);
        
        pp.get("ncomp", ncomp);
        pp.get("nghost", nghost);
        pp.get("periodicity", piv);
        pp.get("nbins", nbins);

        pp.query("name", name);
        pp.query("scaling", scaling);
    }

    Box domain(IntVect{0}, (d_size -= 1));
    BoxArray ba(domain);
    ba.maxSize(mgs);

       int nitems = ba.size();
    amrex::Print() << "Number of buckets: " << nbins << std::endl;
    amrex::Print() << "Number of boxes: " << nitems << std::endl;

    // BUILD WEIGHT DISTRIBUTION AND SORTING BYTES VECTOR
    std::vector<amrex::Real> wgts(nitems);
    std::vector<Long> bytes;

    Real mean = 100000;
    Real stdev = 4523;
    for (int i = 0; i < nitems; ++i) {
        wgts[i] = amrex::RandomNormal(mean, stdev);
    }

    // Scale weights and convert to Long for algorithms.
    std::vector<Long> scaled_wgts = scale_wgts(wgts);

    // No node_size in serial mode
    std::vector<int> s_dmap = SFC_KnapSackProcessorMapDoIt(ba, scaled_wgts, nbins, &s_eff, true, false, bytes);

    // Print the final distribution map
    amrex::Print() << "Final distribution map:" << std::endl;
    for (size_t i = 0; i < s_dmap.size(); ++i) {
        amrex::Print() << "Box " << i << " -> Rank " << s_dmap[i] << std::endl;
    }
}

// Helper function to scale weights
std::vector<Long> scale_wgts(const std::vector<amrex::Real>& wgts) {
    std::vector<Long> scaled_wgts(wgts.size());
    for (size_t i = 0; i < wgts.size(); ++i) {
        scaled_wgts[i] = static_cast<Long>(wgts[i] * 1e6); // Scale factor for example
    }
    return scaled_wgts;
}


// #include <AMReX.H>
// #include <AMReX_Random.H>
// #include <AMReX_MultiFab.H>
// #include <AMReX_ParmParse.H>
// #include <AMReX_ParallelDescriptor.H>
// #include <random>
// #include <string>
// #include <cstring>  
// #include <omp.h>
// #include <fstream>
// #include <numeric>
// #include <cassert>

// #include <Util.H>
// #include <Knapsack.H>
// #include <SFC.H>
// #include <SFC_knapsack.H>

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
//     int nbins, nnodes, ranks_per_node;
//     IntVect d_size, mgs, nghost, piv;
//     {
//         ParmParse pp;
//         pp.get("domain", d_size);
//         pp.get("max_grid_size", mgs);
//         pp.get("ncomp", ncomp);
//         pp.get("nghost", nghost);
//         pp.get("periodicity", piv);
//         pp.get("nbins", nbins);
//         pp.get("nnodes", nnodes); 
//         pp.get("ranks_per_node", ranks_per_node);  
//         pp.query("name", name);
//         pp.query("scaling", scaling);
//     }


//     Box domain(IntVect{0}, (d_size -= 1));
//     BoxArray ba(domain);
//     ba.maxSize(mgs);

//     int nitems = ba.size();
//     amrex::Print() << "Number of nodes: " << nnodes << std::endl;
//     amrex::Print() << "Number of boxes: " << nitems << std::endl;
//     amrex::Print() << "Ranks per node: " << ranks_per_node << std::endl;

//     std::vector<amrex::Real> wgts(nitems);
//     std::vector<Long> bytes;

//     Real mean = 100000;
//     Real stdev = 4523;
//     for (int i = 0; i < nitems; ++i) {
//         wgts[i] = amrex::RandomNormal(mean, stdev);
//     }
//     std::vector<Long> scaled_wgts = scale_wgts(wgts);

//     amrex::Real sfc_eff = 0.0, knapsack_eff = 0.0;

//     std::vector<int> sfc_knapsack_dmap = SFCProcessorMapDoItCombined(ba, scaled_wgts, nnodes, ranks_per_node, &sfc_eff, &knapsack_eff, true, false, bytes);

//     // Print SFC and Knapsack efficiencies
//     amrex::Print() << "SFC Efficiency: " << sfc_eff << std::endl;
//     amrex::Print() << "Knapsack Efficiency: " << knapsack_eff << std::endl;
// }
