#include <AMReX_BoxArray.H>
#include <AMReX_INT.H>
#include <AMReX_Extension.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>

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

