import os
import sys
import time
import numpy as np
from hnsw_index import HNSWIndex
from annoy import AnnoyIndex
import nmslib
import pickle
from sklearn.neighbors import NearestNeighbors, KDTree

class Index(object):
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def dump_filename(self):
        return self.name + 'pk'

class BruteForceIndex(Index):
    def __init__(self, X):
        if os.path.exists(self.dump_filename):
            with open(self.dump_filename, 'rb') as f:
                self._neigh = pickle.load(f)
        else:
            self._neigh = NearestNeighbors(algorithm='brute')
            self._neigh.fit(X)
            with open(self.dump_filename, 'wb') as f:
                pickle.dump(self._neigh, f)

    def query(self, Y, count):
        dist, ind = self._neigh.kneighbors(Y, count, return_distance=True)
        return ind, dist

class KDTreeIndex(Index):
    def __init__(self, X):
        if os.path.exists(self.dump_filename):
            with open(self.dump_filename, 'rb') as f:
                self._tree = pickle.load(f)
        else:
            self._tree = KDTree(X, leaf_size=2)
            with open(self.dump_filename, 'wb') as f:
                pickle.dump(self._tree, f)

    def query(self, Y, count):
        dist, ind = self._tree.query(Y, count, return_distance=True)
        return ind, dist

class NMSLibIndex(Index):
    def __init__(self, X):
        self._nms_index = nmslib.init(method='hnsw', space='l2')

        if os.path.exists(self.dump_filename):
            self._nms_index.loadIndex(self.dump_filename)
        else:
            self._nms_index.addDataPointBatch(X)
            self._nms_index.createIndex(print_progress=True)
            self._nms_index.saveIndex(self.dump_filename)

    def query(self, Y, count):
        results = self._nms_index.knnQueryBatch(Y, k=count, num_threads=1)
        idxs, dists = [], []
        for x, y in results:
            idxs.append(x)
            dists.append(y)
        return np.array(idxs), np.array(dists)

class AnnoyGenIndex(Index):
    def __init__(self, X):
        self._annoy_index = AnnoyIndex(len(X[0]), metric='euclidean')
        if os.path.exists(self.dump_filename):
            self._annoy_index.load(self.dump_filename)
        else:
            for i, embedding in enumerate(X):
                self._annoy_index.add_item(i, embedding)
            self._annoy_index.build(10)
            self._annoy_index.save(self.dump_filename)

    def query(self, Y, count):
        idxs, dists = [], []
        for y in Y:
            idx, dist = self._annoy_index.get_nns_by_vector(y, count, include_distances=True)
            idxs.append(idx)
            dists.append(dist)
        return np.array(idxs), np.array(dists)

class HNSWCustomIndex(Index):
    def __init__(self, X):
        self._hnsw_index = HNSWIndex(len(X[0]))
        if os.path.exists(self.dump_filename):
            self._hnsw_index.load_index(self.dump_filename)
        else:
            self._hnsw_index.add_items(X, np.arange(len(X)))
            self._hnsw_index.save_index(self.dump_filename)

    def query(self, Y, count):
        best_matches_idxs, distances = self._hnsw_index.knn_query(Y, k=count)
        return best_matches_idxs, distances

def generate_data(train_size=1000000, test_size=100):
    dump_filename = 'XY.pk'
    if os.path.exists(dump_filename):
        with open(dump_filename, 'rb') as f:
            X, Y = pickle.load(f)
    else:
        data_dim = 128
        X = np.random.random((train_size, data_dim))
        Y = np.random.random((test_size, data_dim))
        with open(dump_filename, 'wb') as f:
            pickle.dump([X, Y], f)
    return X, Y

def get_mean_best_position(true_matrix, pred_matrix):
    def best_position(true_idxs, pred_idxs):
        best_idx = true_idxs[0]
        pred_ranks = {idx: i for i, idx in enumerate(pred_idxs)}
        return pred_ranks.get(best_idx, len(pred_matrix))
    best_positions = [best_position(true_idxs, pred_idxs) for true_idxs, pred_idxs in zip(true_matrix, pred_matrix)]
    return np.mean(best_positions)

def get_mean_chosen_position(true_matrix, pred_matrix):
    def chosen_position(true_idxs, pred_idxs):
        chosen_idx = pred_idxs[0]
        true_ranks = {idx: i for i, idx in enumerate(true_idxs)}
        return true_ranks.get(chosen_idx, len(true_idxs))
    chosen_positions = [chosen_position(true_idxs, pred_idxs) for true_idxs, pred_idxs in zip(true_matrix, pred_matrix)]
    return np.mean(chosen_positions)

# mean number of embeddings in true 10 closest
def get_kof10(true_matrix, pred_matrix):
    def get_countof10(true_idxs, pred_idxs):
        first_true_idxs = set(true_idxs[:10])
        return len([idx for idx in pred_idxs[:10] if idx in first_true_idxs])
    chosen_positions = [get_countof10(true_idxs, pred_idxs) for true_idxs, pred_idxs in zip(true_matrix, pred_matrix)]
    return np.mean(chosen_positions)

def sort_indexes_by_distance(idxs, distances):
    result_idxs = []
    for idxs_line, dists in zip(idxs, distances):
        sorted_idxs = [x[0] for x in sorted(zip(idxs_line, dists), key=lambda x: x[1])]
        result_idxs.append(sorted_idxs)
    return np.array(result_idxs)

def test():
    X, Y = generate_data(train_size=100000, test_size=100)
    print('data sizes: train: {}, test: {}'.format(len(X), len(Y)))
    cls_indexes = [BruteForceIndex, KDTreeIndex, NMSLibIndex, AnnoyGenIndex, HNSWCustomIndex]
    brutforce_index = None
    indexes = []
    # init
    for clas in cls_indexes:
        indexes.append(clas(X))
        if brutforce_index is None: # indexes[0]
            brutforce_index = indexes[-1]

    print('--- search quality ---')
    true_indxs, distances = brutforce_index.query(Y, 100000)
    true_indxs = sort_indexes_by_distance(true_indxs, distances)
    for k in [1, 5, 10, 100]:
        for index in indexes[1:]:
            idxs, distances = index.query(Y, k)
            if distances is not None:
                idxs = sort_indexes_by_distance(idxs, distances)
            line_to_show = '{}\tk = {}\t'.format(index.name, k)

            mean1 = get_mean_best_position(true_indxs, idxs)
            line_to_show += 'mean_best_pos:\t{}\t'.format(mean1)

            mean2 = get_mean_chosen_position(true_indxs, idxs)
            line_to_show += 'mean_chos_pos:\t{}\t'.format(mean2)

            mean3 = get_kof10(true_indxs, idxs)
            line_to_show += 'mean_k_of_10:\t{}'.format(mean3)
            print(line_to_show)

if __name__ == '__main__':
    test()
