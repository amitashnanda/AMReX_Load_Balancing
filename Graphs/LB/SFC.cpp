#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_Morton.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelContext.H>

#include <SFC.H>
#include <Knapsack.H>
#include <LeastUsed.H>

#if 0
AMREX_FORCE_INLINE
bool
SFCToken::Compare::operator () (const SFCToken& lhs,
                                const SFCToken& rhs) const
{
#if (AMREX_SPACEDIM == 1)
        return lhs.m_morton[0] < rhs.m_morton[0];
#elif (AMREX_SPACEDIM == 2)
        return (lhs.m_morton[1] <  rhs.m_morton[1]) ||
              ((lhs.m_morton[1] == rhs.m_morton[1]) &&
               (lhs.m_morton[0] <  rhs.m_morton[0]));
#else
        return (lhs.m_morton[2] <  rhs.m_morton[2]) ||
              ((lhs.m_morton[2] == rhs.m_morton[2]) &&
              ((lhs.m_morton[1] <  rhs.m_morton[1]) ||
              ((lhs.m_morton[1] == rhs.m_morton[1]) &&
               (lhs.m_morton[0] <  rhs.m_morton[0]))));
#endif
}
#endif

namespace {

    AMREX_FORCE_INLINE
    SFCToken makeSFCToken (int box_index, amrex::IntVect const& iv)
    {
        SFCToken token;
        token.m_box = box_index;

#if (AMREX_SPACEDIM == 3)

        constexpr int imin = -(1 << 29);
        AMREX_ASSERT_WITH_MESSAGE(AMREX_D_TERM(iv[0] >= imin && iv[0] < -imin,
                                            && iv[1] >= imin && iv[1] < -imin,
                                            && iv[2] >= imin && iv[2] < -imin),
                                  "SFCToken: index out of range");
        uint32_t x = iv[0] - imin;
        uint32_t y = iv[1] - imin;
        uint32_t z = iv[2] - imin;
        // extract lowest 10 bits and make space for interleaving
        token.m_morton[0] = amrex::Morton::makeSpace(x & 0x3FF)
                         | (amrex::Morton::makeSpace(y & 0x3FF) << 1)
                         | (amrex::Morton::makeSpace(z & 0x3FF) << 2);
        x = x >> 10;
        y = y >> 10;
        z = z >> 10;
        token.m_morton[1] = amrex::Morton::makeSpace(x & 0x3FF)
                         | (amrex::Morton::makeSpace(y & 0x3FF) << 1)
                         | (amrex::Morton::makeSpace(z & 0x3FF) << 2);
        x = x >> 10;
        y = y >> 10;
        z = z >> 10;
        token.m_morton[2] = amrex::Morton::makeSpace(x & 0x3FF)
                         | (amrex::Morton::makeSpace(y & 0x3FF) << 1)
                         | (amrex::Morton::makeSpace(z & 0x3FF) << 2);

#elif (AMREX_SPACEDIM == 2)

        constexpr uint32_t offset = 1u << 31;
        static_assert(static_cast<uint32_t>(std::numeric_limits<int>::max())+1 == offset,
                      "INT_MAX != (1<<31)-1");
        uint32_t x = (iv[0] >= 0) ? static_cast<uint32_t>(iv[0]) + offset
            : static_cast<uint32_t>(iv[0]-std::numeric_limits<int>::lowest());
        uint32_t y = (iv[1] >= 0) ? static_cast<uint32_t>(iv[1]) + offset
            : static_cast<uint32_t>(iv[1]-std::numeric_limits<int>::lowest());
        // extract lowest 16 bits and make sapce for interleaving
        token.m_morton[0] = amrex::Morton::makeSpace(x & 0xFFFF)
                         | (amrex::Morton::makeSpace(y & 0xFFFF) << 1);
        x = x >> 16;
        y = y >> 16;
        token.m_morton[1] = amrex::Morton::makeSpace(x) | (amrex::Morton::makeSpace(y) << 1);

#elif (AMREX_SPACEDIM == 1)

        constexpr uint32_t offset = 1u << 31;
        static_assert(static_cast<uint32_t>(std::numeric_limits<int>::max())+1 == offset,
                      "INT_MAX != (1<<31)-1");
        token.m_morton[0] = (iv[0] >= 0) ? static_cast<uint32_t>(iv[0]) + offset
            : static_cast<uint32_t>(iv[0]-std::numeric_limits<int>::lowest());

#else
        static_assert(false,"AMREX_SPACEDIM != 1, 2 or 3");
#endif

        return token;
    }
}

void
Distribute (const std::vector<SFCToken>&     tokens,
            const std::vector<amrex::Long>&  wgts,
            int                              nprocs,
            amrex::Real                      volpercpu,
            std::vector< std::vector<int> >& v,
            bool                             flag_verbose_mapper)

{
    BL_PROFILE("Distribute()");

    if (flag_verbose_mapper) {
        amrex::Print() << "Distribute:" << std::endl;
        amrex::Print() << "  volpercpu: " << volpercpu << std::endl;
        amrex::Print() << "  Sorted SFC Tokens:" << std::endl;
        int idx = 0;
        for (const auto &t : tokens) {
            amrex::Print() << "    " << idx++ << ": "
                           << t.m_box << ": "
                           << t.m_morton << std::endl;
        }
    }

    BL_ASSERT(static_cast<int>(v.size()) == nprocs);

    int  K        = 0;
    amrex::Real totalvol = 0;

    for (int i = 0; i < nprocs; ++i)
    {
        int  cnt = 0;
        amrex::Real vol = 0;

        for ( int TSZ = static_cast<int>(tokens.size());
              K < TSZ && (i == (nprocs-1) || (vol < volpercpu));
              ++K)
        {
            vol += wgts[tokens[K].m_box];
            ++cnt;

            v[i].push_back(tokens[K].m_box);
        }

        totalvol += vol;

        if ((totalvol/(i+1)) > volpercpu &&  // Too much for this bin.
            cnt > 1                      &&  // More than one box in this bin.
            i < nprocs-1)                    // Not the last bin, which has to take all.
        {
            --K;
            v[i].pop_back();
            totalvol -= wgts[tokens[K].m_box];
        }
    }

    if (flag_verbose_mapper) {
        amrex::Print() << "Distributed SFC Tokens:" << std::endl;
        int idx = 0;
        for (int i = 0; i < nprocs; ++i) {
            amrex::Print() << "  Rank/Team " << i << ":" << std::endl;
            amrex::Real rank_vol = 0;
            for (const auto &box : v[i]) {
                amrex::ignore_unused(box);
                const auto &t = tokens[idx];
                BL_ASSERT(box == t.m_box);
                amrex::Print() << "    " << idx << ": "
                               << t.m_box << ": "
                               << t.m_morton << std::endl;
                rank_vol += wgts[t.m_box];
                idx++;
            }
            amrex::Print() << "    Total Rank Vol: " << rank_vol << std::endl;
        }
    }

#ifdef AMREX_DEBUG
    int cnt = 0;
    for (int i = 0; i < nprocs; ++i) {
        cnt += v[i].size();
    }
    BL_ASSERT(cnt == static_cast<int>(tokens.size()));
#endif
}

std::vector<int>
SFCProcessorMapDoIt (const amrex::BoxArray&          boxes,
                     const std::vector<amrex::Long>& wgts,
                     int                             nprocs,
                     amrex::Real*                    eff,
                     int                             node_size,
                     bool                            flag_verbose_mapper,
                     bool                            sort,
                     const std::vector<amrex::Long>& bytes)

{
    if (flag_verbose_mapper) {
        amrex::Print() << "DM: SFCProcessorMapDoIt called..." << std::endl;
    }

    BL_PROFILE("SFCProcessorMapDoIt()");

    int nteams = nprocs;
    int nworkers = 1;
#if defined(BL_USE_TEAM)
    nteams = ParallelDescriptor::NTeams();
    nworkers = ParallelDescriptor::TeamSize();
#else
    if (node_size > 0) {
        nteams = nprocs/node_size;
        nworkers = node_size;
        if (nworkers*nteams != nprocs) {
            nteams = nprocs;
            nworkers = 1;
        }
    }
#endif

    if (flag_verbose_mapper) {
        amrex::Print() << "  (nprocs, nteams, nworkers) = ("
                       << nprocs << ", " << nteams << ", " << nworkers << ")\n";
    }

    const int N = boxes.size();
    std::vector<SFCToken> tokens;
    tokens.reserve(N);
    for (int i = 0; i < N; ++i)
    {
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

    std::vector< std::vector<int> > vec(nteams);

    Distribute(tokens,wgts,nteams,volperteam,vec);

    // vec has a size of nteams and vec[] holds a vector of box ids.

    tokens.clear();

    std::vector<LIpair> LIpairV;

    LIpairV.reserve(nteams);

    for (int i = 0; i < nteams; ++i)
    {
        amrex::Long wgt = 0;
        const std::vector<int>& vi = vec[i];
        for (int j = 0, M = vi.size(); j < M; ++j)
            wgt += wgts[vi[j]];

        LIpairV.push_back(LIpair(wgt,i));
    }

    if (sort) Sort(LIpairV, true);

    if (flag_verbose_mapper) {
        for (const auto &p : LIpairV) {
            amrex::Print() << "  Bucket " << p.second << " contains " << p.first << std::endl;
        }
    }

    // LIpairV has a size of nteams and LIpairV[] is pair whose first is weight
    // and second is an index into vec.  LIpairV is sorted by weight such that
    // LIpairV is the heaviest.

    amrex::Vector<int> ord;
    amrex::Vector<amrex::Vector<int> > wrkerord;

    if (nteams == nprocs) {
        if (sort) {
            LeastUsedCPUs(nprocs, bytes, ord, flag_verbose_mapper);
        } else {
            ord.resize(nprocs);
            std::iota(ord.begin(), ord.end(), 0);
        }
    } else {
        if (sort) {
//            LeastUsedTeams(ord,wrkerord,nteams,nworkers);
        } else {
            ord.resize(nteams);
            std::iota(ord.begin(), ord.end(), 0);
            wrkerord.resize(nteams);
            for (auto& v : wrkerord) {
                v.resize(nworkers);
                std::iota(v.begin(), v.end(), 0);
            }
        }
    }

    // ord is a vector of process (or team) ids, sorted from least used to more heavily used.
    // wrkerord is a vector of sorted worker ids.

    std::vector<int> result(wgts.size());

    for (int i = 0; i < nteams; ++i)
    {
        const int tid  = ord[i];                  // tid is team id
        const int ivec = LIpairV[i].second;       // index into vec
        const std::vector<int>& vi = vec[ivec];   // this vector contains boxes assigned to this team
        const int Nbx = vi.size();                // # of boxes assigned to this team

        // if (flag_verbose_mapper) {
        //     amrex::Print() << "Mapping bucket " << LIpairV[i].second << " to rank " << ord[i] << std::endl;
        // }

        if (nteams == nprocs) { // In this case, team id is process id.
            for (int j = 0; j < Nbx; ++j)
            {
                result[vi[j]] = amrex::ParallelContext::local_to_global_rank(tid);
//                m_ref->m_pmap[vi[j]] = amrex::ParallelContext::local_to_global_rank(tid);
            }
        }
        else   // We would like to do knapsack within the team workers
        {
            std::vector<amrex::Long> local_wgts;
            for (int j = 0; j < Nbx; ++j) {
                local_wgts.push_back(wgts[vi[j]]);
            }

            std::vector<std::vector<int> > kpres;
            amrex::Real kpeff;
            knapsack(local_wgts, nworkers, kpres, kpeff, true, N);

            // kpres has a size of nworkers. kpres[] contains a vector of indices into vi.

            // sort the knapsacked chunks
            std::vector<LIpair> ww;
            for (int w = 0; w < nworkers; ++w) {
                amrex::Long wgt = 0;
                for (std::vector<int>::const_iterator it = kpres[w].begin();
                     it != kpres[w].end(); ++it)
                {
                    wgt += local_wgts[*it];
                }
                ww.push_back(LIpair(wgt,w));
            }
            Sort(ww,true);

            // ww is a sorted vector of pair whose first is the weight and second is a index
            // into kpres.

            const amrex::Vector<int>& sorted_workers = wrkerord[i];

            const int leadrank = tid * nworkers;

            for (int w = 0; w < nworkers; ++w)
            {
                const int cpu = leadrank + sorted_workers[w];
                int ikp = ww[w].second;
                const std::vector<int>& js = kpres[ikp];
                for (std::vector<int>::const_iterator it = js.begin(); it!=js.end(); ++it)
                    { result[vi[*it]] = cpu; }
//                    m_ref->m_pmap[vi[*it]] = cpu;
            }
        }
    }

    if (eff || flag_verbose_mapper)
    {
        amrex::Real sum_wgt = 0, max_wgt = 0;
        for (int i = 0; i < nteams; ++i)
        {
            const amrex::Long W = LIpairV[i].first;
            if (W > max_wgt) max_wgt = W;
            sum_wgt += W;
        }
        amrex::Real efficiency = (sum_wgt/(nteams*max_wgt));
        if (eff) *eff = efficiency;

        if (flag_verbose_mapper)
        {
            amrex::Print() << "SFC efficiency: " << efficiency << '\n';
        }
    }

    return result;
}

#if 0

void
SFCProcessorMap (const BoxArray& boxes,
                 int             nprocs)
{
    BL_ASSERT(boxes.size() > 0);

    m_ref->clear();
    m_ref->m_pmap.resize(boxes.size());

    if (boxes.size() < sfc_threshold*nprocs)
    {
        KnapSackProcessorMap(boxes,nprocs);
    }
    else
    {
        std::vector<Long> wgts;

        wgts.reserve(boxes.size());

        for (int i = 0, N = boxes.size(); i < N; ++i)
        {
            wgts.push_back(boxes[i].volume());
        }

        SFCProcessorMapDoIt(boxes,wgts,nprocs);
    }
}

void
SFCProcessorMap (const BoxArray&          boxes,
                 const std::vector<Long>& wgts,
                 int                      nprocs,
                 bool                     sort)
{
    BL_ASSERT(boxes.size() > 0);
    BL_ASSERT(boxes.size() == static_cast<int>(wgts.size()));

    m_ref->clear();
    m_ref->m_pmap.resize(wgts.size());

    if (boxes.size() < sfc_threshold*nprocs)
    {
        KnapSackProcessorMap(wgts,nprocs);
    }
    else
    {
        SFCProcessorMapDoIt(boxes,wgts,nprocs,sort);
    }
}

void
SFCProcessorMap (const BoxArray&          boxes,
                 const std::vector<Long>& wgts,
                 int                      nprocs,
                 Real&                    eff,
                 bool                     sort)
{
    BL_ASSERT(boxes.size() > 0);
    BL_ASSERT(boxes.size() == static_cast<int>(wgts.size()));

    m_ref->clear();
    m_ref->m_pmap.resize(wgts.size());

    if (boxes.size() < sfc_threshold*nprocs)
    {
        KnapSackProcessorMap(wgts,nprocs,&eff);
    }
    else
    {
        SFCProcessorMapDoIt(boxes,wgts,nprocs,sort,&eff);
    }
}

void
RRSFCDoIt (const BoxArray&          boxes,
           int                      nprocs)
{
    BL_PROFILE("DistributionMapping::RRSFCDoIt()");

#if defined (BL_USE_TEAM)
    amrex::Abort("Team support is not implemented yet in RRSFC");
#endif

    const int nboxes = boxes.size();
    std::vector<SFCToken> tokens;
    tokens.reserve(nboxes);
    for (int i = 0; i < nboxes; ++i)
    {
        const Box& bx = boxes[i];
        tokens.push_back(makeSFCToken(i, bx.smallEnd()));
    }
    //
    // Put'm in Morton space filling curve order.
    //
    std::sort(tokens.begin(), tokens.end(), SFCToken::Compare());

    Vector<int> ord;

    LeastUsedCPUs(nprocs,ord);

    // Distribute boxes using roundrobin
    for (int i = 0; i < nboxes; ++i) {
        m_ref->m_pmap[i] = ParallelContext::local_to_global_rank(ord[i%nprocs]);
    }
}

void
RRSFCProcessorMap (const BoxArray&          boxes,
                   int                      nprocs)
{
    BL_ASSERT(boxes.size() > 0);

    m_ref->clear();
    m_ref->m_pmap.resize(boxes.size());

    RRSFCDoIt(boxes,nprocs);
}

void
ComputeDistributionMappingEfficiency (const DistributionMapping& dm,
                                      const Vector<Real>& cost,
                                      Real* efficiency)
{
    const int nprocs = ParallelDescriptor::NProcs();

    // This will store mapping from processor to the costs of FABs it controls,
    // (proc) --> ([cost_FAB_1, cost_FAB_2, ... ]),
    // for each proc
    Vector<Vector<Real>> rankToCosts(nprocs);

    // Count the number of costs belonging to each rank
    Vector<int> cnt(nprocs);
    for (int i=0; i<dm.size(); ++i)
    {
        ++cnt[dm[i]];
    }

    for (int i=0; i<rankToCosts.size(); ++i)
    {
        rankToCosts[i].reserve(cnt[i]);
    }

    for (int i=0; i<cost.size(); ++i)
    {
        rankToCosts[dm[i]].push_back(cost[i]);
    }

    Real maxCost = -1.0;

    // This will store mapping from (proc) --> (sum of cost) for each proc
    Vector<Real> rankToCost(nprocs);
    for (int i=0; i<nprocs; ++i)
    {
        const Real rwSum = std::accumulate(rankToCosts[i].begin(),
                                           rankToCosts[i].end(), 0.0_rt);
        rankToCost[i] = rwSum;
        maxCost = std::max(maxCost, rwSum);
    }

    // Write `efficiency` (number between 0 and 1), the mean cost per processor
    // (normalized to the max cost)
    *efficiency = (std::accumulate(rankToCost.begin(),
                                   rankToCost.end(), 0.0_rt) / (nprocs*maxCost));
}

DistributionMapping
makeSFC (const Vector<Real>& rcost, const BoxArray& ba, bool sort)
{
    BL_PROFILE("makeSFC");

    DistributionMapping r;

    Vector<Long> cost(rcost.size());

    Real wmax = *std::max_element(rcost.begin(), rcost.end());
    Real scale = (wmax == 0) ? 1.e9_rt : 1.e9_rt/wmax;

    for (int i = 0; i < rcost.size(); ++i) {
        cost[i] = Long(rcost[i]*scale) + 1L;
    }

    int nprocs = ParallelContext::NProcsSub();

    r.SFCProcessorMap(ba, cost, nprocs, sort);

    return r;
}

DistributionMapping
makeSFC (const Vector<Real>& rcost, const BoxArray& ba, Real& eff, bool sort)
{
    BL_PROFILE("makeSFC");

    DistributionMapping r;

    Vector<Long> cost(rcost.size());

    Real wmax = *std::max_element(rcost.begin(), rcost.end());
    Real scale = (wmax == 0) ? 1.e9_rt : 1.e9_rt/wmax;

    for (int i = 0; i < rcost.size(); ++i) {
        cost[i] = Long(rcost[i]*scale) + 1L;
    }

    int nprocs = ParallelContext::NProcsSub();

    r.SFCProcessorMap(ba, cost, nprocs, eff, sort);

    return r;
}

DistributionMapping
makeSFC (const LayoutData<Real>& rcost_local,
         Real& currentEfficiency, Real& proposedEfficiency,
         bool broadcastToAll, int root)
{
    BL_PROFILE("makeSFC");

    // Proposed distribution mapping is computed from global vector of costs on root;
    // required information is gathered on root from the layoutData information
    //
    // Two main steps:
    // 1. collect from rcost_local into the global cost vector rcost; then rcost is
    //    complete (only) on root
    // 2. (optional; default true) Broadcast processor map of the new dm to others

    Vector<Real> rcost(rcost_local.size());
    ParallelDescriptor::GatherLayoutDataToVector<Real>(rcost_local, rcost, root);
    // rcost is now filled out on root;

    DistributionMapping r;
    if (ParallelDescriptor::MyProc() == root)
    {
        Vector<Long> cost(rcost.size());

        Real wmax = *std::max_element(rcost.begin(), rcost.end());
        Real scale = (wmax == 0) ? 1.e9_rt : 1.e9_rt/wmax;

        for (int i = 0; i < rcost.size(); ++i) {
            cost[i] = Long(rcost[i]*scale) + 1L;
        }

        // `sort` needs to be false here since there's a parallel reduce function
        // in the processor map function, but we are executing only on root
        int nprocs = ParallelDescriptor::NProcs();
        r.SFCProcessorMap(rcost_local.boxArray(), cost, nprocs, proposedEfficiency, false);

        ComputeDistributionMappingEfficiency(rcost_local.DistributionMap(),
                                             rcost,
                                             &currentEfficiency);
    }

#ifdef BL_USE_MPI
    // Load-balanced distribution mapping is computed on root; broadcast the cost
    // to all proc (optional)
    if (broadcastToAll)
    {
        Vector<int> pmap(rcost_local.DistributionMap().size());
        if (ParallelDescriptor::MyProc() == root)
        {
            pmap = r.ProcessorMap();
        }

        // Broadcast vector from which to construct new distribution mapping
        ParallelDescriptor::Bcast(&pmap[0], pmap.size(), root);
        if (ParallelDescriptor::MyProc() != root)
        {
            r = DistributionMapping(pmap);
        }
    }
#else
    amrex::ignore_unused(broadcastToAll);
#endif

    return r;
}

std::vector<std::vector<int> >
makeSFC (const BoxArray& ba, bool use_box_vol, const int nprocs)
{
    BL_PROFILE("makeSFC");

    const int N = ba.size();
    std::vector<SFCToken> tokens;
    std::vector<Long> wgts;
    tokens.reserve(N);
    wgts.reserve(N);
    Long vol_sum = 0;
    for (int i = 0; i < N; ++i)
    {
        const Box& bx = ba[i];
        tokens.push_back(makeSFCToken(i, bx.smallEnd()));
        const Long v = use_box_vol ? bx.volume() : Long(1);
        vol_sum += v;
        wgts.push_back(v);
    }
    //
    // Put'm in Morton space filling curve order.
    //
    std::sort(tokens.begin(), tokens.end(), SFCToken::Compare());

    Real volper;
    volper = vol_sum / nprocs;

    std::vector< std::vector<int> > r(nprocs);

    Distribute(tokens, wgts, nprocs, volper, r);

    return r;
}

#endif
