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

    size_t max_links = 32;
    size_t search_horizon_size = 100;
    InsertStrategy insert_method = InsertStrategy::link_diverse;
};



template<class Key,
        class Vector,
        class Distance,
        class Random = std::minstd_rand>
class Index {
public:
    using Scalar = decltype(std::declval<Distance>()(std::declval<Vector>(), std::declval<Vector>()));

    struct SearchResult {
        Key key;
        Scalar distance;

        bool operator<(const SearchResult &other) const {
            return this->distance < other.distance;
        }

        bool operator>(const SearchResult &other) const {
            return this->distance > other.distance;
        }

        std::pair<Key, Scalar> ToPair() const {
            return {key, distance};
        };
    };

    IndexOptions options;
    Distance distance;
    Random random;

private:
    struct Node {
        /* Contains links between keys in the Hierarchical Navigable Small World graph. */
        using OutEdges = flat_map<Key, Scalar>;
        using InEdges = std::set<Key>;

        struct LayeredEdges {
            /* Contains connections to nodes on the same layer. */
            OutEdges out_edges;
            InEdges in_edges;
        };

        Vector vector;
        std::vector<LayeredEdges> layers;
    };

    std::unordered_map<Key, Node> nodes;
    std::map<size_t, std::unordered_set<Key>> levels; // some levels might be empty

    using ClosestFirstQueue = std::priority_queue<SearchResult,
            std::vector<SearchResult>, std::greater<>
    >;

    using FurthestFirstQueue = std::priority_queue<SearchResult,
            std::vector<SearchResult>, std::less<>
    >;

public:
    void insert(const Key &key, const Vector &vector) {
        insert(key, Vector(vector));
    }

    void insert(const Key &key, Vector &&vector) {
        if (nodes.count(key) > 0) {
            throw std::runtime_error("Index::insert <- Key already exists!");
        }

        size_t node_level = random_level();

        // create a Node for the key
        auto node_it = nodes.emplace(key, Node {
                std::move(vector),
                std::vector<typename Node::LayeredEdges>(node_level)
        }).first;
        Node& node = node_it->second;

        // initialize layers of the node
        for (size_t layer = 0; layer < node_level; ++layer) {
            node.layers[layer].out_edges.reserve(max_links(layer));
        }

        // if the level is empty — put the key into it
        if (nodes.size() == 1) {
            levels[node_level].insert(key);
            return;
        }


        Key current_search_key = *levels.rbegin()->second.begin(); // picks a node from the highest level
        for (size_t layer = nodes[current_search_key].layers.size(); layer > 0; --layer) {
            current_search_key = greedy_search(node.vector, layer - 1, current_search_key);

            if (layer <= node_level) {
                detail::SequenceAccessQueue<FurthestFirstQueue> results;
                search_level(node.vector,
                             options.search_horizon_size * 2,
                             layer - 1,
                             {current_search_key},
                             results);

                std::sort(results.c.begin(), results.c.end());
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


    std::vector<SearchResult> search(const Vector &target, size_t nearest_neighbors) const {
        return search(target, nearest_neighbors, options.search_horizon_size + nearest_neighbors);
    }


    std::vector<SearchResult> search(const Vector &target, size_t nearest_neighbors, size_t ef) const {
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
                results.c.end()
        );

        std::vector<SearchResult> results_vector(results_to_return);
        for (size_t i = 0; i < results_to_return; ++i) {
            results_vector[i] = {results.c[i].key, results.c[i].distance};
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
                const auto &links = node.second.layers[layer].out_edges;

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

                    if (peer_node_it->second.layers[layer].in_edges.count(node.first) == 0) {
                        return false;
                    }
                }

                for (const auto &link: node.second.layers[layer].in_edges) {
                    auto peer_node_it = nodes.find(link);

                    if (peer_node_it == nodes.end()) {
                        return false;
                    }

                    if (layer >= peer_node_it->second.layers.size()) {
                        return false;
                    }

                    if (peer_node_it->second.layers[layer].out_edges.count(node.first) == 0) {
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
        double sample = random();
        double normalized_sample = (sample - Random::min()) / (Random::max() - Random::min());

        double level_multiplier = 1. / std::log(options.max_links);

        auto level = -std::log(normalized_sample) * level_multiplier;
        return static_cast<size_t>(level) + 1;
    }


    void search_level(const Vector &target,
                      size_t n_results,
                      size_t layer,
                      const std::vector<Key> &current_front,
                      FurthestFirstQueue &results) const {
        std::unordered_set<Key> visited_nodes(current_front.begin(), current_front.end());

        detail::SequenceAccessQueue<ClosestFirstQueue> search_front;
        for (const Key &key: current_front) {
            auto distance_to_target = distance(target, nodes.at(key).vector);
            results.push({key, distance_to_target});
            search_front.push({key, distance_to_target});
        }

        while (results.size() > n_results) {
            results.pop();
        }

        for (size_t hop = 0; !search_front.empty() &&
                             (search_front.top().distance <= results.top().distance) &&
                             (hop < nodes.size()); ++hop) {
            const Node &current_node = nodes.at(search_front.top().key);
            const auto &links = current_node.layers[layer].out_edges;
            search_front.pop();


            for (const auto &link: links) {
                const Key& link_key = link.first;
                bool is_link_unvisited = visited_nodes.insert(link_key).second;

                if (is_link_unvisited) {
                    auto link_distance = distance(target, nodes.at(link_key).vector);

                    if (results.size() < n_results) {
                        results.push({link_key, link_distance});
                        search_front.push({link_key, link_distance});
                    } else if (link_distance < results.top().distance) {
                        results.pop();
                        results.push({link_key, link_distance});
                        search_front.push({link_key, link_distance});
                    }
                }
            }

            // Try to make search_front smaller, so to speed up operations on it.
            while (!search_front.empty() && (search_front.c.back().distance > results.top().distance)) {
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
            bool is_hop_made = false;

            const auto &links = node.layers[layer].out_edges;

            for (const auto &link : links) {
                const Key& link_key = link.first;
                Scalar neighbor_distance = distance(target, nodes.at(link_key).vector);

                if (neighbor_distance < result_distance) {
                    result = link_key;
                    result_distance = neighbor_distance;
                    is_hop_made = true;
                }
            }

            if (!is_hop_made) {
                break;
            }
        }

        return result;
    }

    void try_add_link(const Key &node,
                      size_t layer,
                      const Key &new_link,
                      Scalar link_distance) {
        auto &layer_links = nodes[node].layers[layer].out_edges;

        if (layer_links.size() < max_links(layer)) {
            layer_links.emplace(new_link, link_distance);
            nodes[new_link].layers[layer].in_edges.insert(node);
            return;
        }

        if (options.insert_method == IndexOptions::InsertStrategy::link_nearest) {
            auto furthest_key = layer_links.begin()->first;
            auto furthest_distance = layer_links.begin()->second;

            for (const auto& candidate : layer_links) {
                auto& key = candidate.first;
                auto& distance = candidate.second;
                if (key == new_link) {
                    return;
                }

                if (distance > furthest_distance) {
                    furthest_key = key;
                    furthest_distance = distance;
                }
            }

            if (link_distance < furthest_distance) {
                layer_links.erase(furthest_key);
                nodes[furthest_key].layers[layer].in_edges.erase(node);
                layer_links.emplace(new_link, link_distance);
                nodes[new_link].layers[layer].in_edges.insert(node);
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
        const auto &new_link_vector = nodes[new_link].vector;

        for (const auto &link: sorted_links) {
            if (link.first == new_link) {
                insert = false;
                break;
            }
        }

        if (insert) {
            for (size_t i = 0; i < sorted_links.size(); ++i) {
                const Key &current_node = sorted_links[i].first;
                const Scalar &current_distance = sorted_links[i].second;

                if (link_distance >= current_distance) {
                    if (link_distance > distance(new_link_vector, nodes[current_node].vector)) {
                        insert = false;
                        break;
                    }
                } else if (replace_index > i) {
                    if (current_distance > distance(new_link_vector, nodes[current_node].vector)) {
                        replace_index = i;
                    }
                }
            }
        }

        if (insert) {
            const Key& replacement_candidate_key = sorted_links[replace_index].first;
            nodes[replacement_candidate_key].layers[layer].in_edges.erase(node);
            nodes[new_link].layers[layer].in_edges.insert(node);
            layer_links.erase(replacement_candidate_key);
            layer_links.emplace(new_link, link_distance);
        }
    }


    // new_links_set - *sorted by distance to the node* sequence of unique elements
    void set_links(const Key &node,
                   size_t layer,
                   const std::vector<SearchResult> &new_links_set) {
        size_t required_links_count = max_links(layer);
        std::vector<SearchResult> new_links;
        new_links.reserve(required_links_count);

        if (options.insert_method == IndexOptions::InsertStrategy::link_nearest) {
            new_links.assign(
                    new_links_set.begin(),
                    new_links_set.begin() + std::min(new_links_set.size(), required_links_count)
            );
        } else {
            select_diverse_links(max_links(layer), new_links_set, new_links);
        }

        auto &outgoing_links = nodes[node].layers[layer].out_edges;

        for (const auto &link: outgoing_links) {
            nodes[link.first].layers[layer].in_edges.erase(node);
        }

        std::sort(new_links.begin(), new_links.end(), [](const auto &l, const auto &r) { return l.key < r.key; });
        std::vector<std::pair<Key, Scalar>> transformed_links(new_links.size());
        std::transform(new_links.begin(), new_links.end(),
                       transformed_links.begin(),
                       [](const SearchResult& entry) {return entry.ToPair();});
        outgoing_links.assign_ordered_unique(transformed_links.begin(), transformed_links.end());

        for (const auto &entry: new_links) {
            nodes[entry.key].layers[layer].in_edges.insert(node);
        }
    }


    void select_diverse_links(size_t links_number,
                              const std::vector<SearchResult> &candidates,
                              std::vector<SearchResult> &result) const {
        std::vector<const Vector *> links_vectors;
        links_vectors.reserve(links_number);

        std::vector<SearchResult> rejected;
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

};

}

int main() {
    auto index = hnsw::Index<u_int32_t, std::vector<float>, hnsw::l2_square_distance_t>();

    index.options.insert_method = hnsw::IndexOptions::InsertStrategy::link_diverse;
    for (u_int32_t i = 0; i < 10; ++i) {
        auto key = i;
        auto vector = std::vector<float> {
            float(i), float(i % 3), 0
        };
        index.insert(key, vector);
    }
    auto vector_to_search = std::vector<float> {
            0, 2, 0
    };
    auto query = index.search(vector_to_search, 5);


    for (const auto& result : query) {
        std::cout << result.key << " " << result.distance << std::endl;
    }

    std::cout << "Check complete: " << (index.check() ? "True" : "False") << std::endl;
    return 0;
}