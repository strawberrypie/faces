#include <omp.h>

namespace hnsw {

namespace detail {

template<class T>
T l2sqr_dist(const T *one, const T* another, size_t size) {
    T sum = 0;

    #pragma omp simd
    for (size_t i = 0; i < size; ++i) {
        auto diff = one[i] - another[i];
        sum += diff * diff;
    }

    return sum;
}

}

struct l2_square_distance_t {

    template<class Vector>
    auto operator()(const Vector &one, const Vector &another) const {
        if (one.size() != another.size()) {
            throw std::runtime_error("l2_square_distance_t: vectors sizes do not match");
        }

        return detail::l2sqr_dist(one.data(), another.data(), one.size());
    }
};
}