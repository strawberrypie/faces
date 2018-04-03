

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
class ValuesAccessQueue : public Base {
public:
    using Base::c;
    typename Base::container_type &values = c;
};

}

}