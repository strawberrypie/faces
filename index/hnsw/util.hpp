namespace hnsw {
namespace util {

struct ClosestResultComparator {
    template<typename T>
    bool operator()(const T &l, const T &r) const {
        return l.second < r.second;
    }
};


struct FurthestResultComparator {
    template<typename T>
    bool operator()(const T &l, const T &r) const {
        return l.second > r.second;
    }
};


template<typename Base>
class ValuesAccessQueue : public Base {
public:
    using Base::c;
    typename Base::container_type &values = c;
};

}
}