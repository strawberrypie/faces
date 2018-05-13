#include <random>
#include "index.hpp"


using Vector = std::vector<float>;
using Distance = hnsw::L2SquareDistance;
using Key = u_int32_t;
using Index = hnsw::Index<Key, Vector, Distance>;
using LinearIndex = std::vector<std::pair<Key, Vector>>;


static Vector generate_unit_sphere_vector(size_t size) {
    /* Generates a vector uniformly distributed on a unit sphere.
     * Source: https://mathoverflow.net/a/24690
     * */
    static std::normal_distribution<float> normal(0, 1);
    static std::default_random_engine generator;

    Vector data(size);
    std::generate(data.begin(), data.end(),
                  []() { return normal(generator); });

    float norm = 0.; // calculate norm
    for (size_t i = 0; i < size; ++i) {
        norm += data[i] * data[i];
    }
    norm = std::sqrt(norm);

    for (size_t i = 0; i < size; ++i) { // normalize
        data[i] /= norm;
    }
    return data;
}


std::vector<Index::SearchResult> linear_search(
        const Vector &target,
        LinearIndex &linear_index,
        size_t n_neighbors = 3) {
    static Distance dist;

    std::sort(linear_index.begin(), linear_index.end(),
              [&target](const std::pair<Key, Vector> &left,
                        const std::pair<Key, Vector> &right) {
                  return dist(left.second, target) < dist(right.second, target);
              });

    std::vector<Index::SearchResult> result(n_neighbors);
    for (size_t i = 0; i < std::min(n_neighbors, linear_index.size()); ++i) {
        result[i] = {linear_index[i].first, dist(linear_index[i].second, target)};
    }
    return result;
}


int main() {
    size_t n_dim = 300;
    size_t n_vectors = 10000;

    auto index = hnsw::Index<Key, Vector, Distance>();
    LinearIndex linear_index;

    for (u_int32_t key = 0; key < n_vectors; ++key) {
        auto vector = generate_unit_sphere_vector(n_dim);
        index.insert(key, vector);
        linear_index.emplace_back(std::make_pair(key, vector));
    }
    std::cout << "Indexes created!\n" << std::endl;

    auto target_vector = generate_unit_sphere_vector(n_dim);

    std::cout << "HNSW index results:" << std::endl;
    auto query = index.search(target_vector, 5);
    for (const auto &result : query) {
        std::cout << result.key << " " << result.distance << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Linear index results:" << std::endl;
    auto linear_query = linear_search(target_vector, linear_index, 5);
    for (const auto &result : linear_query) {
        std::cout << result.key << " " << result.distance << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Check passed: " << (index.check() ? "True" : "False") << std::endl;
    return 0;
}
