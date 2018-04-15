#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/unordered_set.hpp>


#include "distance.hpp"
#include "util.hpp"


namespace hnsw {

struct IndexOptions {
    size_t max_links = 32;
    size_t search_horizon_size = 100;
};

template<typename Key,
        typename Vector,
        typename Distance,
        typename Random = std::minstd_rand>
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

    Random random;
    Distance distance;
    IndexOptions options;

private:
    struct Node {
        /* Contains links between keys in the Hierarchical Navigable Small World graph. */
        using OutEdges = std::unordered_map<Key, Scalar>;
        using InEdges = std::set<Key>;

        struct LayeredEdges {
            /* Contains connections to nodes on the same layer. */
            OutEdges out_edges;
            InEdges in_edges;

        private:
            friend class cereal::access;

            template<class Archive>
            void serialize(Archive &ar) {
                ar(out_edges, in_edges);
            }
        };

        Vector vector;
        std::vector<LayeredEdges> layers;

    private:
        friend class cereal::access;

        template<class Archive>
        void serialize(Archive &ar) {
            ar(vector, layers);
        }
    };

    std::unordered_map<Key, Node> nodes;
    std::map<size_t, std::unordered_set<Key>> levels; // some levels might be empty

    using ClosestFirstQueue = std::priority_queue<SearchResult,
            std::vector<SearchResult>, std::greater<>
    >;

    using FurthestFirstQueue = std::priority_queue<SearchResult,
            std::vector<SearchResult>, std::less<>
    >;

    friend class cereal::access;

    template<class Archive>
    void serialize(Archive &ar) {
        ar(nodes, levels);
    }
public:
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

                    if (peer_node_it->second.layers.at(layer).in_edges.count(node.first) == 0) {
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

                    if (peer_node_it->second.layers.at(layer).out_edges.count(node.first) == 0) {
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

    void insert(const Key &key, const Vector &vector) {
        insert(key, Vector(vector));
    }

    void insert(const Key &key, Vector &&vector) {
        if (nodes.count(key) > 0) {
            throw std::runtime_error("index::insert <- Key already exists!");
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
            // for upper layers — search for the closer node
            current_search_key = greedy_search(node.vector, current_search_key, layer - 1);

            // for lower layers — connect to neighbors found by doing a search
            if (layer <= node_level) {
                util::ValuesAccessQueue<FurthestFirstQueue> neighbors;
                search_level(node.vector, options.search_horizon_size * 2, layer - 1,
                             {current_search_key}, neighbors);

                std::sort(neighbors.values.begin(), neighbors.values.end());
                connect_set_exclusively(key, neighbors.values, layer - 1);

                for (const auto &peer: neighbors.values) {
                    connect(peer.key, key, peer.distance, layer - 1);
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
            start = greedy_search(target, start, layer - 1);
        }

        util::ValuesAccessQueue<FurthestFirstQueue> results;
        search_level(target, std::max(nearest_neighbors, ef), 0, {start}, results);

        size_t results_to_return = std::min(results.size(), nearest_neighbors);
        std::partial_sort(
                results.values.begin(),
                results.values.begin() + results_to_return,
                results.values.end()
        );

        std::vector<SearchResult> results_vector(results_to_return);
        for (size_t i = 0; i < results_to_return; ++i) {
            results_vector[i] = {results.values[i].key, results.values[i].distance};
        }
        return results_vector;
    }


    void save_index(const std::string &path) {
        std::ofstream out_file(path, std::ios::binary);
        {
            cereal::BinaryOutputArchive archive(out_file);
            archive(*this);
        }
    }


    void load_index(const std::string &path) {
        std::ifstream in_file(path);
        {
            cereal::BinaryInputArchive archive(in_file);
            archive(*this);
        }
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

        // initialize the search front
        util::ValuesAccessQueue<ClosestFirstQueue> search_front;
        for (const Key &key: current_front) {
            auto distance_to_target = distance(target, nodes.at(key).vector);
            results.push({key, distance_to_target});
            search_front.push({key, distance_to_target});
        }

        // main loop of the search
        for (size_t hop = 0; !search_front.empty() &&
                             (search_front.top().distance <= results.top().distance) &&
                             (hop < nodes.size()); ++hop) {
            // pick a node
            const Node &current_node = nodes.at(search_front.top().key);
            search_front.pop();
            const auto &links = current_node.layers[layer].out_edges;

            // extend the front with closer neighbors
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

            // reduce the front size
            while (!search_front.empty() && (search_front.values.back().distance > results.top().distance)) {
                search_front.values.pop_back();
            }
        }

        // leave only the required number of results
        while (results.size() > n_results) {
            results.pop();
        }
    }


    Key greedy_search(const Vector &target, const Key &start_from, size_t layer) const {
        /* Searches greedily for a closest node in index to target vector.
         * Starts from `start_from`, and works on the level `layer`.*/
        Key result = start_from;
        Scalar result_distance = distance(target, nodes.at(start_from).vector);

        /* Let's hop to a closer neighbor while we can. */
        for (size_t hops = 0; hops < nodes.size(); ++hops) {
            const Node &node = nodes.at(result);
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


    void connect(const Key &node_from, const Key &node_to, Scalar link_distance, size_t layer) {
        /* Creates a bidirectional link between node_from and node_to. */
        auto &out_edges = nodes[node_from].layers[layer].out_edges;

        // if there is space for new links
        if (out_edges.size() < max_links(layer)) {
            // connect bidirectionally
            out_edges.emplace(node_to, link_distance);
            nodes[node_to].layers[layer].in_edges.insert(node_from);
            return;
        }

        // otherwise keep only max_links(layer) connections to the closest nodes to node_from
        auto furthest_key = out_edges.begin()->first;
        auto furthest_distance = out_edges.begin()->second;

        for (const auto& candidate : out_edges) {
            auto& key = candidate.first;
            auto& distance = candidate.second;
            if (key == node_to) {
                return;
            }

            if (distance > furthest_distance) {
                furthest_key = key;
                furthest_distance = distance;
            }
        }

        if (link_distance < furthest_distance) {
            out_edges.erase(furthest_key);
            nodes[furthest_key].layers[layer].in_edges.erase(node_from);
            out_edges.emplace(node_to, link_distance);
            nodes[node_to].layers[layer].in_edges.insert(node_from);
        }
    }

    
    void connect_set_exclusively(const Key &node, const std::vector<SearchResult> &links_set, size_t layer) {
        /* Replaces the connections from node with connections to links_set. */
        auto &out_edges = nodes[node].layers[layer].out_edges;
        for (const auto &link: out_edges) {
            nodes[link.first].layers[layer].in_edges.erase(node);
        }

        std::vector<SearchResult> new_links(
                links_set.begin(),
                links_set.begin() + std::min(links_set.size(), max_links(layer))
        );
        std::sort(new_links.begin(), new_links.end(), [](const auto &l, const auto &r) { return l.key < r.key; });
        std::vector<std::pair<Key, Scalar>> transformed_links(new_links.size());
        std::transform(new_links.begin(), new_links.end(),
                       transformed_links.begin(),
                       [](const SearchResult& entry) {return entry.ToPair();});
        out_edges.clear();
        out_edges.insert(transformed_links.begin(), transformed_links.end());

        for (const auto &entry: new_links) {
            nodes[entry.key].layers[layer].in_edges.insert(node);
        }
    }
};

}
