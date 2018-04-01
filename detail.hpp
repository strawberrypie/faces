

namespace hnsw {

namespace detail {

struct ClosestResultComparator {
    template<class T>
    bool operator()(const T &l, const T &r) const {
        return l.second < r.second;
    }
};


struct FurthestResultComparator {
    template<class T>
    bool operator()(const T &l, const T &r) const {
        return l.second > r.second;
    }
};


template<class Base>
class SequenceAccessQueue : public Base {
public:
    using base_type = Base;
    using Base::c;
};

}

}