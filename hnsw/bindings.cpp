#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Index.cpp"

namespace py = pybind11;

using KeyType = uint32_t;
using VectorType = std::vector<float>;
using DistanceType = hnsw::CosineDistance;
using IndexType = hnsw::Index<KeyType, VectorType, DistanceType>;


class HNSWIndex {
public:
    HNSWIndex(const size_t dim) : dim(dim), index() {}

    void addItems(py::object _vectors, py::object _keys) {
        py::array_t<float, py::array::c_style | py::array::forcecast> vectors(_vectors);
        auto buffer = vectors.request();

        size_t rows = buffer.shape[0];
        size_t features = buffer.shape[1];

        std::vector<size_t> ids;
        py::array_t<size_t, py::array::c_style | py::array::forcecast> keys(_keys);

        size_t keys_count = keys.request().shape[0];
        if (keys_count != rows) {
            throw std::runtime_error("the number of rows and keys should be equal");
        }
        if (features != dim) {
            throw std::runtime_error("Wrong dimensions: index has " + std::to_string(dim) +
                                     " while got " + std::to_string(features));
        }

        for (size_t i = 0; i < rows; ++i) {
            VectorType vector(features);
            for (size_t j = 0; j < features; ++j){
                vector[j] = *vectors.data(i, j);
            }
            index.insert(*keys.data(i), vector);
        }
    }

    py::object knnQuery_return_numpy(py::object _vectors, size_t k = 1) {

        py::array_t<float, py::array::c_style | py::array::forcecast> vectors(_vectors);
        auto buffer = vectors.request();
        size_t rows =  buffer.shape[0];
        size_t features = buffer.shape[1];

        if (buffer.ndim != 2) throw std::runtime_error("data must be a 2d array");

        std::vector<size_t> result_keys(rows * k);
        std::vector<float> result_distances(rows * k);
        for (size_t i = 0; i < rows; ++i) {
            VectorType vector(features);
            for (size_t j = 0; j < features; ++j){
                vector[j] = *vectors.data(i, j);
            }
            auto search_result = index.search(vector, k);
            for (size_t j = 0; j < k; ++j) {
                result_keys[k * i + j] = search_result[j].key;
                result_distances[k * i + j] = search_result[j].key;
            }
        }

        py::capsule free_when_done_keys(result_keys.data(), [](void *f) {
            delete[] f;
        });

        py::capsule free_when_done_distances(result_distances.data(), [](void *f) {
            delete[] f;
        });

        return py::make_tuple(
                py::array_t<size_t>(
                        {rows, k},
                        {k * sizeof(size_t), sizeof(size_t)},
                        result_keys.data(),
                        free_when_done_keys),
                py::array_t<float>(
                        {rows, k},
                        {k * sizeof(float), sizeof(float)},
                        result_distances.data(),
                        free_when_done_distances)
                );
    }

    IndexType index;
    size_t dim;
};

using BridgeType = HNSWIndex;

PYBIND11_PLUGIN(hnsw) {
    py::module m("hnsw");

    py::class_<BridgeType>(m, "HNSWIndex")
            .def(py::init<const int>(), py::arg("dim"))
            .def("knn_query", &BridgeType::knnQuery_return_numpy, py::arg("data"), py::arg("k")=1)
            .def("add_items", &BridgeType::addItems, py::arg("data"), py::arg("ids"))
            .def("__repr__",
                 [](const BridgeType &a) {
                     return "<HNSW-lib index>";
                 }
            );
    return m.ptr();
}