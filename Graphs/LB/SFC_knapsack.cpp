#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_Morton.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelContext.H>

#include <SFC.H>
#include <Knapsack.H>
#include <LeastUsed.H>
#include <SFC_knapsack.H>


// Define the SFC and Knapsack functions here
std::vector<int>
SFCProcessorMapDoItCombined (const amrex::BoxArray&          boxes,
                     const std::vector<amrex::Long>& wgts,
                     int                             nnodes,
                     int                             ranks_per_node,
                     amrex::Real*                    sfc_eff,
                     amrex::Real*                    knapsack_eff,
                     bool                            flag_verbose_mapper,
                     bool                            sort,
                     const std::vector<amrex::Long>& bytes)

{
    if (flag_verbose_mapper) {
        amrex::Print() << "DM: SFCProcessorMapDoItCombined called..." << std::endl;
    }

    BL_PROFILE("SFCProcessorMapDoItCombined()");

    // RUN SFC with "node" number of bins 

    const int nteams = nnodes;
    // int nteams = nnodes;
//     int nworkers = 1;

// #if defined(BL_USE_TEAM)
//     nteams = ParallelDescriptor::NTeams();
//     nworkers = ParallelDescriptor::TeamSize();     

// #else
//     if (node_size > 0) {
//         nteams = nnodes/node_size;
//         nworkers = node_size;
//         if (nworkers*nteams != nnodes) {
//             nteams = nnodes;
//             nworkers = 1;
//         }
//     } 
// #endif
    if (flag_verbose_mapper) {
        amrex::Print() << "  (nnodes, nteams, ranks_per_node) = ("
                       << nnodes << ", " << nteams << ", " << ranks_per_node << ")\n";
    }

    const int N = boxes.size();
    std::vector<SFCToken> tokens;
    tokens.reserve(N);
    for (int i = 0; i < N; ++i) {
        const amrex::Box& bx = boxes[i];
        tokens.push_back(makeSFCToken(i, bx.smallEnd()));
    }


    //
    // Put'm in Morton space filling curve order.
    //
    std::sort(tokens.begin(), tokens.end(), SFCToken::Compare());
    //
    // Split'm up as equitably as possible per team.
    //
    amrex::Real volperteam = 0;
    for (amrex::Long wt : wgts) {
        volperteam += wt;
    }
    volperteam /= nteams;

    std::vector<std::vector<int>> vec(nteams);

    Distribute(tokens, wgts, nteams, volperteam, vec, flag_verbose_mapper); //// calling SFC 1

/////
    // amrex::Print()<<"Printing SFC vec" << "\n"; 


    // for (int i = 0; i < vec.size(); ++i)
    // {
    //     for (int j = 0; j<vec[i].size();++j)
    //     {
    //         amrex::Print()<<vec[i][j] << "," ;
    //     }
    //     amrex::Print()<<"\n";

    // }
    // amrex::Print()<<"\n\n";  

    ///// Commented Out ////////





    // amrex::Print() << "SFC Distribution Map (Node -> Boxes):\n";



    // for (int i = 0; i < vec.size(); ++i)
    // {
    //     amrex::Print() << "Node " << i << ": ";
    //     for (int j = 0; j < vec[i].size(); ++j)
    //     {
    //         amrex::Print() << vec[i][j];
    //         if (j < vec[i].size() - 1) {
    //             amrex::Print() << ", ";  
    //         }
    //     }
    //     amrex::Print() << "\n";
    // }
    // amrex::Print() << "\n";






    ///// Commented Out ////////

/////

    // vec has a size of nteams and vec[] holds a vector of box ids.

    tokens.clear();

    std::vector<LIpair> LIpairV;
    LIpairV.reserve(nteams);

    for (int i = 0; i < nteams; ++i) {
        amrex::Long wgt = 0;
        const std::vector<int>& vi = vec[i];
        for (int j = 0, M = vi.size(); j < M; ++j)
            wgt += wgts[vi[j]];

        LIpairV.push_back(LIpair(wgt, i));
    }

    if (sort) Sort(LIpairV, true);

    if (flag_verbose_mapper) {
        for (const auto& p : LIpairV) {
            amrex::Print() << "  Bucket " << p.second << " contains " << p.first << std::endl;
        }
    }

    // LIpairV has a size of nteams and LIpairV[] is pair whose first is weight
    // and second is an index into vec. LIpairV is sorted by weight such that
    // LIpairV is the heaviest.


    // This creates the solution vector and initializes it with -1 (bad data)

    std::vector<int> result(wgts.size(), -1);  // Initialize solution with -1 (step 1)

    amrex::Real sum_wgt_sfc = 0, max_wgt_sfc = 0;
    for (int i = 0; i < nteams; ++i) {
        const amrex::Long W = LIpairV[i].first;
        if (W > max_wgt_sfc) max_wgt_sfc = W;
        sum_wgt_sfc += W;
    }
    *sfc_eff = (sum_wgt_sfc / (nteams * max_wgt_sfc)); /// SFC eff

    amrex::Real total_weight_knapsack = 0;
    amrex::Real max_weight_knapsack_across_ranks = 0;



    for (int i = 0; i < nteams; ++i) {

        const int tid = i; // tid is team id
        const int ivec = LIpairV[i].second; // index into vec
        const std::vector<int>& vi = vec[ivec]; // this vector contains boxes assigned to this team
        const int Nbx = vi.size(); // # of boxes assigned to this team

        /// Create lists to store the weights and indices assigned to this node

        std::vector<amrex::Long> local_wgts;
        std::vector<int> local_indices;

        for (int j = 0; j < Nbx; ++j) {
            local_wgts.push_back(wgts[vi[j]]);
            local_indices.push_back(vi[j]); /// Track global index
        }


        /// Print local weights and indices for debugging





        // amrex::Print() << "Node " << i << " Weights and Indices:\n";
        // for (int j = 0; j < Nbx; ++j) {
        //     amrex::Print() << "  Index " << local_indices[j] << " -> Weight " << local_wgts[j] << "\n";
        // }
        // amrex::Print() << "\n";





        // The Knapsack algorithm is run on the smaller weight vector.

        std::vector<std::vector<int>> knapsack_result;
        amrex::Real knapsack_local_efficiency;
        // lowercase knapsack is just like distribute in SFC

        knapsack(local_wgts, ranks_per_node, knapsack_result, knapsack_local_efficiency, true, N);
        amrex::Print() << "Node " << i << " Each Knapsack efficiency: " << knapsack_local_efficiency << "\n";        

        


   



        // *knapsack_eff = knapsack_local_efficiency;




      

        // amrex::Print()<<"Printing Knapsack vec team" << i << "\n"; 


        // for (int i = 0; i < knapsack_result.size(); ++i)
        // {
        //     for (int j = 0; j<knapsack_result[i].size();++j)
        //     {
        //         amrex::Print()<<knapsack_result[i][j] << "," ;
        //     }
        //     amrex::Print()<<"\n";

        // }
        // amrex::Print()<<"\n\n";  

        /// Print the knapsack result with corresponding global indices for each node




        // amrex::Print() << "Knapsack result for Node " << i << ":\n";
        // for (int j = 0; j < knapsack_result.size(); ++j) {
        //     amrex::Print() << "  Processor Group " << j << ": ";
        //     for (int k = 0; k < knapsack_result[j].size(); ++k) {
        //         int global_idx = local_indices[knapsack_result[j][k]];
        //         amrex::Print() << "Global Index " << global_idx << " (Local Weight Index " << knapsack_result[j][k] << "), ";
        //     }
        //     amrex::Print() << "\n";
        // }
        // amrex::Print() << "\n";





        ///

        // The Knapsack results are transferred back into the full solution vector,
        // adjusting the indices to account for the node and rank.

        // int knapsack_idx = 0;
        // for (int j = 0; j < Nbx; ++j) {
        //     assert(result[vi[j]] == -1);  // Ensure the box hasn't already been assigned
        //     result[vi[j]] = knapsack_result[knapsack_idx % ranks_per_node][knapsack_idx / ranks_per_node] + (tid * ranks_per_node);
        //     knapsack_idx++;
        // }

        for (int j = 0; j < knapsack_result.size(); ++j) {
            amrex::Real local_knapsack_wgt = 0;
            for (int k = 0; k < knapsack_result[j].size(); ++k) {

                /// Here, the global index is obtained using local_indices.

                int global_idx = local_indices[knapsack_result[j][k]];
                // assert(result[global_idx] == -1);  /// Ensure the box hasn't already been assigned
                // result[global_idx] = j + (tid * ranks_per_node); //// Map local rank (0-3) to global rank
                // Calculate local weight for each rank
                local_knapsack_wgt += wgts[global_idx];
                int global_rank = (tid * ranks_per_node) + j;
                result[global_idx] = global_rank;
            //     amrex::Print() << "Global Index: " << global_idx << ", Local Rank: " << j
            //    << ", Global Rank: " << global_rank << "\n";
            }

            // Update total and max weights after knapsack redistribution

            if (local_knapsack_wgt > max_weight_knapsack_across_ranks) {
            max_weight_knapsack_across_ranks = local_knapsack_wgt;
            }
            total_weight_knapsack += local_knapsack_wgt;
            
        }

        //Print local results after each node's knapsack run

        // amrex::Print() << "Node " << i << " has total knapsack weight: " << total_weight_knapsack << "\n";
        // amrex::Print() << "Node " << i << " current max weight across ranks: " << max_weight_knapsack_across_ranks << "\n";


    }

    


    

    // amrex::Print()<<"Printing final result" << "\n"; 
    // for (int i = 0; i < result.size(); ++i)
    // {
                
    //     amrex::Print()<<result[i] << "," ;
    // }

    //  amrex::Print()<<"\n\n"; 
             

//// Ensure all boxes have been assigned

    for (int i = 0; i < result.size(); ++i) {
        assert(result[i] != -1);  
    }

    *knapsack_eff = total_weight_knapsack / (ranks_per_node * nteams * max_weight_knapsack_across_ranks);


     




    //// old way of calculating efficiency

    // amrex::Real sum_wgt_knapsack = 0, max_wgt_knapsack = 0;
    // for (int i = 0; i < nteams; ++i) {
    //     amrex::Real local_sum_wgt = 0;
    //     for (const auto& idx : vec[i]) {
    //         local_sum_wgt += wgts[idx];
    //     }
    //     if (local_sum_wgt > max_wgt_knapsack) max_wgt_knapsack = local_sum_wgt;
    //     sum_wgt_knapsack += local_sum_wgt;
    // }
    // *knapsack_eff = (sum_wgt_knapsack / (nteams * max_wgt_knapsack));
    
   /////


    // for (int i=0; i<result.size(); ++i) {
    //     amrex::Print()<<result[i]<<" , ";
    // }
    // amrex::Print()<<"\n";


    if (flag_verbose_mapper) {
        amrex::Print() << "SFC efficiency for combined algorithm: " << *sfc_eff << '\n';
        amrex::Print() << "SFC+Knapsack combined efficiency: " << *knapsack_eff << '\n';
    }



    // Output the distribution map with weights to a CSV file
    std::ofstream outfile("distribution_map_sfc_knapsack.csv");
    outfile << "BoxID,Processor,Weight\n";
    for (size_t i = 0; i < result.size(); ++i) {
        outfile << i << "," << result[i] << "," << wgts[i] << "\n";
    }
    outfile.close();

    return result;
}


/////Things to do

// Built in thing  node_size check this!! compare to my solution
// test with different box sizes
// SFC alone what is the efficiency
// Each Knapsack calculate each time it goes through find the knapsack efficiency
// To see which one is worst node how it impacts the total SFC solution
// Time profile for each knapsack vs SFC need to test 
// double the node and double the dimension for boxes!! go up to 128 nodes 
// inside the knapsack calculate each run efficiency



