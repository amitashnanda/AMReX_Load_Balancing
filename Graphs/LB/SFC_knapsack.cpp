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
        amrex::Print() << "DM: SFCProcessorMapDoIt called..." << std::endl;
    }

    BL_PROFILE("SFCProcessorMapDoIt()");

    // RUN SFC with "node" number of bins 

    const int nteams = nnodes;

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

    Distribute(tokens, wgts, nteams, volperteam, vec, flag_verbose_mapper);



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
    *sfc_eff = (sum_wgt_sfc / (nteams * max_wgt_sfc));

    for (int i = 0; i < nteams; ++i) {
        const int tid = i; // tid is team id
        const int ivec = LIpairV[i].second; // index into vec
        const std::vector<int>& vi = vec[ivec]; // this vector contains boxes assigned to this team
        const int Nbx = vi.size(); // # of boxes assigned to this team

        // For each node, we extract the weights assigned to that node into local_wgts.

        std::vector<amrex::Long> local_wgts;
        for (int j = 0; j < Nbx; ++j) {
            local_wgts.push_back(wgts[vi[j]]);
        }


        // The Knapsack algorithm is run on the smaller weight vector.

        std::vector<std::vector<int>> knapsack_result;
        amrex::Real knapsack_local_efficiency;
        knapsack(local_wgts, ranks_per_node, knapsack_result, knapsack_local_efficiency, true, N);

        // The Knapsack results are transferred back into the full solution vector,
        // adjusting the indices to account for the node and rank.

        int knapsack_idx = 0;
        for (int j = 0; j < Nbx; ++j) {
            assert(result[vi[j]] == -1);  // Ensure the box hasn't already been assigned
            result[vi[j]] = knapsack_result[knapsack_idx % ranks_per_node][knapsack_idx / ranks_per_node] + (tid * ranks_per_node);
            knapsack_idx++;
        }
    }
//// Ensure all boxes have been assigned

    for (int i = 0; i < result.size(); ++i) {
        assert(result[i] != -1);  
    }

    amrex::Real sum_wgt_knapsack = 0, max_wgt_knapsack = 0;
    for (int i = 0; i < nteams; ++i) {
        amrex::Real local_sum_wgt = 0;
        for (const auto& idx : vec[i]) {
            local_sum_wgt += wgts[idx];
        }
        if (local_sum_wgt > max_wgt_knapsack) max_wgt_knapsack = local_sum_wgt;
        sum_wgt_knapsack += local_sum_wgt;
    }
    *knapsack_eff = (sum_wgt_knapsack / (nteams * max_wgt_knapsack));

    if (flag_verbose_mapper) {
        amrex::Print() << "SFC efficiency: " << *sfc_eff << '\n';
        amrex::Print() << "Knapsack efficiency: " << *knapsack_eff << '\n';
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
