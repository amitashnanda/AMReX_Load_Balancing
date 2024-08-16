#ifndef AMREX_SFC
#define AMREX_SFC

#include <AMReX_BoxArray.H>
#include <AMReX_INT.H>
#include <AMReX_Extension.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_Morton.H>


struct SFCToken
{
    class Compare
    {
    public:
        AMREX_FORCE_INLINE
        bool operator () (const SFCToken& lhs,
                          const SFCToken& rhs) const;
    };
    int m_box;
    amrex::Array<uint32_t,AMREX_SPACEDIM> m_morton;
};

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

void
Distribute (const std::vector<SFCToken>&     tokens,
            const std::vector<amrex::Long>&  wgts,
            int                              nprocs,
            amrex::Real                      volpercpu,
            std::vector< std::vector<int> >& v,
            bool                             flag_verbose_mapper = false);

std::vector<int>
SFCProcessorMapDoIt (const amrex::BoxArray&          boxes,
                     const std::vector<amrex::Long>& wgts,
                     int                             nprocs,
                     amrex::Real*                    eff,
                     int                             node_size = 0,
                     bool                            flag_verbose_mapper = false,
                     bool                            sort = false,
                     const std::vector<amrex::Long>& bytes = std::vector<amrex::Long>());

#endif