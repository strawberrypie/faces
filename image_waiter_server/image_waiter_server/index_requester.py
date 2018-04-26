import requests
import os
import logging
import time
import json
import random
import numpy as np
from requests.exceptions import ConnectionError

logging.basicConfig(level=logging.INFO)

class IndexRequester(object):
    def __init__(self, url='http://0.0.0.0', port=8080, retry_count=3, resend_timeout=3):
        self._uri = os.path.join(url + ':' + str(port))
        self._best_matches_path = 'best_matches'
        self._retry_count = retry_count
        self._resend_timeout = resend_timeout

    def make_single_post_request(self, method_name, payload):
        url = os.path.join(self._uri, method_name)
        logging.info('request to url: {}'.format(url))
        try:
            r = requests.post(url, json=payload)
        except ConnectionError as e:
            logging.error('gor: {}'.format(e))
            return None, 500
        if r.status_code == 200:
            return r.json(), r.status_code
        return None, r.status_code

    def make_request(self, request_name, method_name, payload=None):
        retry_count = 0
        while retry_count < self._retry_count:
            # TODO: add choice: get, post, put, delete
            response, status_code = self.make_single_post_request(method_name, payload)
            if status_code == 200:
                return response
            if status_code == 500:
                retry_count += 1
                logging.warning("can't make request, status code: {}, retry after {} sec".format(status_code, self._resend_timeout))
                time.sleep(self._resend_timeout)
            else:
                logging.error('during {} operation got code: {}'.format(method_name, status_code))
                break
        return None

    def get_best_idxs(self, embedding, count):
        payload = {'embedding': [np.float64(x) for x in embedding], 'count': count}
        response = self.make_request('POST', self._best_matches_path, payload)
        if response is None:
            return None, None
        best_idxs, distances = [], []
        for indexes in response['result']:
            best_idxs.append(indexes['index'])
            distances.append(indexes['distance'])
        return best_idxs, distances


class DummyIndexRequester(IndexRequester):
    def __init__(self):
        pass

    def get_best_idxs(self, embedding, count):
        idxs = [x for x in range(10)]
        random.shuffle(idxs)
        idxs = idxs[:count]
        return idxs, [0 for _ in idxs]
