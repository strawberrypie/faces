import numpy as np
import random
import logging
from annoy import AnnoyIndex

ANNOY_FILEPATH = '/media/roman/Other/celebA/tmp/index.ann'
# ANNOY_FILEPATH = '/usr/tmp/index.ann'

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
            logging.info('embedding has wrong size')
            return [], []
        idxs = self._annoy_index.get_nns_by_vector(embedding, count)
        distances = [0 for x in idxs]
        return idxs, distances

class DummyIndexSearch(IndexSearch):
    def get_best_matches(self, embedding, count):
        idxs = [x for x in range(10)]
        random.shuffle(idxs)
        idxs = idxs[:count]
        distances = [0 for _ in idxs]
        return idxs, distances

def test():
    annoy_index = AnnoyIndexSearch()
    idxs, _ = annoy_index.get_best_matches([x for x in range(128)], 2)
    print('idxs', idxs)

if __name__ == '__main__':
    test()
