import numpy as np
import random
import logging
from annoy import AnnoyIndex
from hnsw_index import HNSWIndex

DEBUG = False

if DEBUG:
    ANNOY_FILEPATH = '/media/roman/Other/celebA/tmp/index.ann'
    HNSW_FILEPATH = '/media/roman/Other/celebA/tmp/index.hnsw'
else:
    ANNOY_FILEPATH = '/usr/tmp/index.ann'
    HNSW_FILEPATH = '/usr/tmp/index.hnsw'


class IndexSearch(object):
    def get_best_matches(self, embedding, count):
        raise NotImplemented

class AnnoyIndexSearch(IndexSearch):
    def __init__(self, annoy_size=128, annoy_filepath=ANNOY_FILEPATH):
        self._annoy_size = annoy_size
        self._annoy_index = AnnoyIndex(annoy_size)
        self._annoy_index.load(annoy_filepath)

    def get_best_matches(self, embedding, count):
        if len(embedding) != self._annoy_size:
            logging.warning('embedding has wrong size')
            return [], []
        idxs = self._annoy_index.get_nns_by_vector(embedding, count)
        distances = [0 for x in idxs]
        return idxs, distances

class HNSWIndexSearch(IndexSearch):
    def __init__(self, hnsw_size=128, hnsw_filepath=HNSW_FILEPATH):
        self._hnsw_size = hnsw_size
        self._hnsw_index = HNSWIndex(hnsw_size)
        self._hnsw_index.load_index(hnsw_filepath)

    def get_best_matches(self, embedding, count):
        if len(embedding) != self._hnsw_size:
            logging.warning('embedding has wrong size')
            return [], []
        idxs, distances = self._hnsw_index.knn_query(np.array([embedding]), k=count)
        idxs = [int(x) for x in idxs[0]]
        distances = [float(x) for x in distances[0]]
        return idxs, distances

class DummyIndexSearch(IndexSearch):
    def get_best_matches(self, embedding, count):
        idxs = [x for x in range(10)]
        random.shuffle(idxs)
        idxs = idxs[:count]
        distances = [0 for _ in idxs]
        return idxs, distances

def test():
    index = AnnoyIndexSearch()
    idxs, _ = index.get_best_matches([x for x in range(128)], 2)
    print('idxs', idxs)
    index = HNSWIndexSearch()
    idxs, distances = index.get_best_matches([x for x in range(128)], 2)
    print('idxs', idxs)
    print('distances', distances)

if __name__ == '__main__':
    test()
