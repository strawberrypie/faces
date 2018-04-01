#include <random>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cassert>
#include <xmmintrin.h>
#include "flat_map.hpp"
#include "distance.hpp"
#include "detail.hpp"

namespace hnsw {

struct IndexOptions {
    enum class InsertStrategy {
        link_nearest,
        link_diverse
    };

    enum class RemoveStrategy {
        no_link,
        compensate_incoming_links
    };

    size_t max_links = 32;
    size_t ef_construction = 200;
    InsertStrategy insert_method = InsertStrategy::link_diverse;
    RemoveStrategy remove_method = RemoveStrategy::compensate_incoming_links;
};



template<class Key,
        class Vector,
        class Distance,
        class Random = std::minstd_rand>
struct hnsw_index {
    using Scalar = decltype(std::declval<Distance>()(std::declval<Vector>(), std::declval<Vector>()));

    struct SearchQueryResult {
        Key key;
        Scalar distance;
    };

    struct Node {
        using outgoing_links_t = flat_map<Key, Scalar>;
        using incoming_links_t = std::set<Key>;

        struct LayerConnections {
            outgoing_links_t outgoing;
            incoming_links_t incoming;
        };

        Vector vector;
        std::vector<LayerConnections> layers;
    };


    IndexOptions options;
    Distance distance;
    Random random;

    std::unordered_map<Key, Node> nodes;

    // For levels order of keys is important, so it's std::map.
    std::map<size_t, std::unordered_set<Key>> levels;

private:
    struct SearchEntry {
        Key key;
        Scalar distance;

        bool operator<(const SearchEntry &other) const {
            return this->distance < other.distance;
        }

        bool operator>(const SearchEntry &other) const {
            return this->distance > other.distance;
        }

        std::pair<Key, Scalar> ToPair() const {
            return {key, distance};
        };
    };

    using ClosestFirstQueue = std::priority_queue<SearchEntry,
            std::vector<SearchEntry>, std::greater<>
    >;

    using FurthestFirstQueue = std::priority_queue<SearchEntry,
            std::vector<SearchEntry>, std::less<>
    >;

public:
    void insert(const Key &key, const Vector &vector) {
        insert(key, Vector(vector));
    }

    void insert(const Key &key, Vector &&vector) {
        if (nodes.count(key) > 0) {
            throw std::runtime_error("hnsw_index::insert <- Key already exists!");
        }

        size_t node_level = random_level() + 1;

        auto node_it = nodes.emplace(key, Node {
                std::move(vector),
                std::vector<typename Node::LayerConnections>(node_level)
        }).first;
        auto& node = node_it->second;


        for (size_t layer = 0; layer < node_level; ++layer) {
            node.layers[layer].outgoing.reserve(max_links(layer));
        }

        if (nodes.size() == 1) {
            levels[node_level].insert(key);
            return;
        }

        Key start = *levels.rbegin()->second.begin();

        for (size_t layer = nodes.at(start).layers.size(); layer > 0; --layer) {
            start = greedy_search(node.vector, layer - 1, start);

            if (layer <= node_level) {
                detail::SequenceAccessQueue<FurthestFirstQueue> results;
                search_level(node.vector,
                             options.ef_construction,
                             layer - 1,
                             {start},
                             results);

                std::sort(results.c.begin(), results.c.end(), std::less<>());
                set_links(key, layer - 1, results.c);

                // NOTE: Here we attempt to link all candidates to the new item.
                // The original HNSW attempts to link only with the actual neighbors.
                for (const auto &peer: results.c) {
                    try_add_link(peer.key, layer - 1, key, peer.distance);
                }
            }
        }

        levels[node_level].insert(key);
    }


    void remove(const Key &key) {
        auto node_it = nodes.find(key);

        if (node_it == nodes.end()) {
            return;
        }

        const auto &layers = node_it->second.layers;

        for (size_t layer = 0; layer < layers.size(); ++layer) {
            for (const auto &link: layers[layer].outgoing) {
                nodes.at(link.first).layers.at(layer).incoming.erase(key);
            }

            for (const auto &link: layers[layer].incoming) {
                nodes.at(link).layers.at(layer).outgoing.erase(key);
            }
        }

        if (options.remove_method != IndexOptions::RemoveStrategy::no_link) {
            for (size_t layer = 0; layer < layers.size(); ++layer) {
                for (const auto &inverted_link: layers[layer].incoming) {
                    auto &peer_links = nodes.at(inverted_link).layers.at(layer).outgoing;
                    const Key *new_link_ptr = nullptr;

                    if (options.insert_method == IndexOptions::InsertStrategy::link_nearest) {
                        new_link_ptr = select_nearest_link(inverted_link, peer_links, layers.at(layer).outgoing);
                    } else if (options.insert_method == IndexOptions::InsertStrategy::link_diverse) {
                        new_link_ptr = select_most_diverse_link(inverted_link, peer_links, layers.at(layer).outgoing);
                    } else {
                        assert(false);
                    }

                    if (new_link_ptr) {
                        auto new_link = *new_link_ptr;
                        auto &new_link_node = nodes.at(new_link);
                        auto d = distance(nodes.at(inverted_link).vector, new_link_node.vector);
                        peer_links.emplace(new_link, d);
                        new_link_node.layers.at(layer).incoming.insert(inverted_link);
                        try_add_link(new_link, layer, inverted_link, d);
                    }
                }
            }
        }

        auto level_it = levels.find(layers.size());

        if (level_it == levels.end()) {
            throw std::runtime_error("hnsw_index::remove: the node is not present in the levels index");
        }

        level_it->second.erase(key);

        // Shrink the hash table when it becomes too sparse
        // (to reduce memory usage and ensure linear complexity for iteration).
        if (4 * level_it->second.load_factor() < level_it->second.max_load_factor()) {
            level_it->second.rehash(size_t(2 * level_it->second.size() / level_it->second.max_load_factor()));
        }

        if (level_it->second.empty()) {
            levels.erase(level_it);
        }

        nodes.erase(node_it);

        if (4 * nodes.load_factor() < nodes.max_load_factor()) {
            nodes.rehash(size_t(2 * nodes.size() / nodes.max_load_factor()));
        }
    }


    std::vector<SearchQueryResult> search(const Vector &target, size_t nearest_neighbors) const {
        return search(target, nearest_neighbors, 100 + nearest_neighbors);
    }


    std::vector<SearchQueryResult> search(const Vector &target, size_t nearest_neighbors, size_t ef) const {
        if (nodes.empty()) {
            return {};
        }

        Key start = *levels.rbegin()->second.begin();

        for (size_t layer = nodes.at(start).layers.size(); layer > 0; --layer) {
            start = greedy_search(target, layer - 1, start);
        }

        detail::SequenceAccessQueue<FurthestFirstQueue> results;
        search_level(target, std::max(nearest_neighbors, ef), 0, {start}, results);

        size_t results_to_return = std::min(results.size(), nearest_neighbors);

        std::partial_sort(
                results.c.begin(),
                results.c.begin() + results_to_return,
                results.c.end(),
                [](const auto &l, const auto &r) {
                    return l.distance < r.distance;
                }
        );

        std::vector<SearchQueryResult> results_vector;
        results_vector.reserve(results_to_return);

        for (size_t i = 0; i < results_to_return; ++i) {
            results_vector.push_back({results.c[i].key, results.c[i].distance});
        }

        return results_vector;
    }


    // Check whether the index satisfies its invariants.
    bool check() const {
        if (nodes.empty()) {
            return levels.empty();
        }

        for (const auto &node: nodes) {
            auto level_it = levels.find(node.second.layers.size());

            if (level_it == levels.end()) {
                return false;
            }

            if (level_it->second.count(node.first) == 0) {
                return false;
            }

            for (size_t layer = 0; layer < node.second.layers.size(); ++layer) {
                const auto &links = node.second.layers[layer].outgoing;

                // Self-links are not allowed.
                if (links.count(node.first) > 0) {
                    return false;
                }

                for (const auto &link: links) {
                    auto peer_node_it = nodes.find(link.first);

                    if (peer_node_it == nodes.end()) {
                        return false;
                    }

                    if (layer >= peer_node_it->second.layers.size()) {
                        return false;
                    }

                    if (peer_node_it->second.layers.at(layer).incoming.count(node.first) == 0) {
                        return false;
                    }
                }

                for (const auto &link: node.second.layers[layer].incoming) {
                    auto peer_node_it = nodes.find(link);

                    if (peer_node_it == nodes.end()) {
                        return false;
                    }

                    if (layer >= peer_node_it->second.layers.size()) {
                        return false;
                    }

                    if (peer_node_it->second.layers.at(layer).outgoing.count(node.first) == 0) {
                        return false;
                    }
                }
            }
        }

        for (const auto &level: levels) {
            for (const auto &key: level.second) {
                auto node_it = nodes.find(key);

                if (node_it == nodes.end()) {
                    return false;
                }

                if (level.first != node_it->second.layers.size()) {
                    return false;
                }
            }
        }

        return true;
    }


private:
    size_t max_links(size_t level) const {
        return (level == 0) ? (2 * options.max_links) : options.max_links;
    }


    size_t random_level() {
        // I avoid use of uniform_real_distribution to control how many times random() is called.
        // This makes inserts reproducible across standard libraries.

        // NOTE: This works correctly for standard random engines because their value_type is required to be unsigned.
        auto sample = random() - Random::min();
        auto max_rand = Random::max() - Random::min();

        // If max_rand is too large, decrease it so that it can be represented by double.
        if (max_rand > 1048576) {
            sample /= max_rand / 1048576;
            max_rand /= max_rand / 1048576;
        }

        double x = std::min(1.0, std::max(0.0, double(sample) / double(max_rand)));
        return static_cast<size_t>(-std::log(x) / std::log(double(options.max_links + 1)));
    }


    void search_level(const Vector &target,
                      size_t results_number,
                      size_t layer,
                      const std::vector<Key> &start_from,
                      FurthestFirstQueue &results) const
    {
        std::unordered_set<Key> visited_nodes;
        visited_nodes.reserve(5 * max_links(layer) * results_number);
        visited_nodes.insert(start_from.begin(), start_from.end());

        detail::SequenceAccessQueue<ClosestFirstQueue> search_front;

        for (const auto &key: start_from) {
            auto d = distance(target, nodes.at(key).vector);
            results.push({key, d});
            search_front.push({key, d});
        }

        while (results.size() > results_number) {
            results.pop();
        }

        for (size_t hop = 0; !search_front.empty() && search_front.top().distance <= results.top().distance && hop < nodes.size(); ++hop) {
            const auto &node = nodes.at(search_front.top().key);
            search_front.pop();

            const auto &links = node.layers.at(layer).outgoing;

//            for (auto it = links.rbegin(); it != links.rend(); ++it) {
//                if (visited_nodes.count(it->first) == 0) {
//                    prefetch<Vector>::pref(nodes.at(it->first).vector);
//                }
//            }

            for (const auto &link: links) {
                if (visited_nodes.insert(link.first).second) {
                    auto d = distance(target, nodes.at(link.first).vector);

                    if (results.size() < results_number) {
                        results.push({link.first, d});
                        search_front.push({link.first, d});
                    } else if (d < results.top().distance) {
                        results.pop();
                        results.push({link.first, d});
                        search_front.push({link.first, d});
                    }
                }
            }

            // Try to make search_front smaller, so to speed up operations on it.
            while (!search_front.empty() && search_front.c.back().distance > results.top().distance) {
                search_front.c.pop_back();
            }
        }
    }


    Key greedy_search(const Vector &target, size_t layer, const Key &start_from) const {
        Key result = start_from;
        Scalar result_distance = distance(target, nodes.at(start_from).vector);

        // Just a reasonable upper limit on the number of hops to avoid infinite loops.
        for (size_t hops = 0; hops < nodes.size(); ++hops) {
            const auto &node = nodes.at(result);
            bool made_hop = false;

            const auto &links = node.layers.at(layer).outgoing;

            for (auto it = links.begin(); it != links.end(); ++it) {
//                if (std::next(it, 1) != links.end()) {
//                    prefetch<Vector>::pref(nodes.at(std::next(it, 1)->first).vector);
//                }

                Scalar neighbor_distance = distance(target, nodes.at(it->first).vector);

                if (neighbor_distance < result_distance) {
                    result = it->first;
                    result_distance = neighbor_distance;
                    made_hop = true;
                }
            }

            if (!made_hop) {
                break;
            }
        }

        return result;
    }

    void try_add_link(const Key &node,
                      size_t layer,
                      const Key &new_link,
                      Scalar link_distance)
    {
        auto &layer_links = nodes.at(node).layers.at(layer).outgoing;

        if (layer_links.size() < max_links(layer)) {
            layer_links.emplace(new_link, link_distance);
            nodes.at(new_link).layers.at(layer).incoming.insert(node);
            return;
        }

        if (options.insert_method == IndexOptions::InsertStrategy::link_nearest) {
            auto furthest_key = layer_links.begin()->first;
            auto furthest_distance = layer_links.begin()->second;

            auto start = layer_links.begin() + 1;
            auto stop = layer_links.end();
            for (auto it = start; it < stop; ++it) {
                if (it->first == new_link) {
                    return;
                }

                if (it->second > furthest_distance) {
                    furthest_key = it->first;
                    furthest_distance = it->second;
                }
            }

            if (link_distance < furthest_distance) {
                layer_links.erase(furthest_key);
                nodes.at(furthest_key).layers.at(layer).incoming.erase(node);
                layer_links.emplace(new_link, link_distance);
                nodes.at(new_link).layers.at(layer).incoming.insert(node);
            }

            return;
        }

        std::vector<std::pair<Key, Scalar>> sorted_links(layer_links.begin(), layer_links.end());

        std::sort(sorted_links.begin(),
                  sorted_links.end(),
                  [](const auto &l, const auto &r) { return l.second < r.second; });

        if (link_distance >= sorted_links.back().second) {
            return;
        }

        bool insert = true;
        size_t replace_index = sorted_links.size() - 1;
        const auto &new_link_vector = nodes.at(new_link).vector;

        for (const auto &link: sorted_links) {
            if (link.first == new_link) {
                insert = false;
                break;
            }
        }

        if (insert) {
            for (size_t i = 0; i < sorted_links.size(); ++i) {
//                if (i + 1 < sorted_links.size()) {
//                    prefetch<Vector>::pref(nodes.at(sorted_links[i + 1].first).vector);
//                }

                if (link_distance >= sorted_links[i].second) {
                    if (link_distance > distance(new_link_vector, nodes.at(sorted_links[i].first).vector)) {
                        insert = false;
                        break;
                    }
                } else if (replace_index > i) {
                    if (sorted_links[i].second > distance(new_link_vector, nodes.at(sorted_links[i].first).vector)) {
                        replace_index = i;
                    }
                }
            }
        }

        if (insert) {
            nodes.at(sorted_links.at(replace_index).first).layers.at(layer).incoming.erase(node);
            nodes.at(new_link).layers.at(layer).incoming.insert(node);
            layer_links.erase(sorted_links.at(replace_index).first);
            layer_links.emplace(new_link, link_distance);
        }
    }


    // new_links_set - *sorted by distance to the node* sequence of unique elements
    void set_links(const Key &node,
                   size_t layer,
                   const std::vector<SearchEntry> &new_links_set)
    {
        size_t need_links = max_links(layer);
        std::vector<SearchEntry> new_links;
        new_links.reserve(need_links);

        if (options.insert_method == IndexOptions::InsertStrategy::link_nearest) {
            new_links.assign(
                    new_links_set.begin(),
                    new_links_set.begin() + std::min(new_links_set.size(), need_links)
            );
        } else {
            select_diverse_links(max_links(layer), new_links_set, new_links);
        }

        auto &outgoing_links = nodes.at(node).layers.at(layer).outgoing;

        for (const auto &link: outgoing_links) {
            nodes.at(link.first).layers.at(layer).incoming.erase(node);
        }

        std::sort(new_links.begin(), new_links.end(), [](const auto &l, const auto &r) { return l.key < r.key; });
        std::vector<std::pair<Key, Scalar>> transformed_links(new_links.size());
        std::transform(new_links.begin(), new_links.end(),
                       transformed_links.begin(),
                       [](const SearchEntry& entry) {return entry.ToPair();});
        outgoing_links.assign_ordered_unique(transformed_links.begin(), transformed_links.end());

        for (const auto &entry: new_links) {
            nodes.at(entry.key).layers.at(layer).incoming.insert(node);
        }
    }


    void select_diverse_links(size_t links_number,
                              const std::vector<SearchEntry> &candidates,
                              std::vector<SearchEntry> &result) const
    {
        std::vector<const Vector *> links_vectors;
        links_vectors.reserve(links_number);

        std::vector<SearchEntry> rejected;
        rejected.reserve(links_number);

        for (const auto &candidate: candidates) {
            if (result.size() >= links_number) {
                break;
            }

            const auto &candidate_vector = nodes.at(candidate.key).vector;
            bool reject = false;

            for (const auto &link_vector: links_vectors) {
                if (distance(candidate_vector, *link_vector) < candidate.distance) {
                    reject = true;
                    break;
                }
            }

            if (reject) {
                if (rejected.size() < links_number) {
                    rejected.push_back(candidate);
                }
            } else {
                result.push_back(candidate);
                links_vectors.push_back(&candidate_vector);
            }
        }

        for (const auto &link: rejected) {
            if (result.size() >= links_number) {
                break;
            }

            result.push_back(link);
        }
    }


    const Key *select_nearest_link(const Key &link_to,
                                     const typename Node::outgoing_links_t &existing_links,
                                     const typename Node::outgoing_links_t &candidates) const
    {
        auto closest_key_it = candidates.end();
        Scalar min_distance = 0;

        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            if (it->first != link_to && existing_links.count(it->first) == 0) {
                auto d = distance(nodes.at(it->first).vector, nodes.at(link_to).vector);

                if (closest_key_it == candidates.end() || d < min_distance) {
                    closest_key_it = it;
                    min_distance = d;
                }
            }
        }

        if (closest_key_it == candidates.end()) {
            return nullptr;
        } else {
            return &closest_key_it->first;
        }
    }


    const Key *select_most_diverse_link(const Key &link_to,
                                          const typename Node::outgoing_links_t &existing_links,
                                          const typename Node::outgoing_links_t &candidates) const
    {
        std::vector<std::pair<const Key *, Scalar>> filtered;
        filtered.reserve(candidates.size());

        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            if (it->first != link_to && existing_links.count(it->first) == 0) {
                filtered.push_back({
                                           &it->first,
                                           distance(nodes.at(link_to).vector, nodes.at(it->first).vector)
                                   });
            }
        }

        std::sort(filtered.begin(),
                  filtered.end(),
                  [](const auto &l, const auto &r) { return l.second < r.second; });

        auto to_insert = filtered.end();

//        for (auto it = existing_links.rbegin(); it != existing_links.rend(); ++it) {
//            prefetch<Vector>::pref(nodes.at(it->first).vector);
//        }

        for (auto candidate_it = filtered.begin(); candidate_it != filtered.end(); ++candidate_it) {
            bool good = true;

            for (auto existing_link = existing_links.begin(); existing_link < existing_links.end(); ++existing_link) {
                auto d = distance(nodes.at(existing_link->first).vector,
                                  nodes.at(*candidate_it->first).vector);

                if (d < candidate_it->second) {
                    good = false;
                    break;
                }
            }

            if (good) {
                to_insert = candidate_it;
                break;
            }
        }

        if (to_insert == filtered.end() && !filtered.empty()) {
            to_insert = filtered.begin();
        }

        if (to_insert == filtered.end()) {
            return nullptr;
        } else {
            return to_insert->first;
        }
    }

};

}

int main() {
    auto index = hnsw::hnsw_index<u_int32_t, std::vector<float>, hnsw::l2_square_distance_t>();

    for (u_int32_t i = 0; i < 10; ++i) {
        auto key = i;
        auto vector = std::vector<float> {
            float(i), float(i % 3), 0
        };
        index.insert(key, vector);
    }
    auto vector_to_search = std::vector<float> {
            6, 0, 0
    };
    auto query = index.search(vector_to_search, 5);


    for (const auto& result : query) {
        std::cout << result.key << " " << result.distance << std::endl;
    }
    return 0;
}