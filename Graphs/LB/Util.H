#include <AMReX_INT.H>
#include <AMReX_REAL.H>
#include <type_traits>

template <class T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
std::vector<amrex::Long>
scale_wgts(std::vector<T> wgts)
{

    double wmax = *std::max_element(wgts.begin(), wgts.end());
    double scale = (wmax == 0) ? double(1e9) : double(1.e9)/wmax;
    std::vector<amrex::Long> swgts(wgts.size());
    for (long unsigned int i = 0; i<wgts.size(); ++i) {
        swgts[i] = amrex::Long(wgts[i]*scale) + 1L;
    }

    return swgts;
}
