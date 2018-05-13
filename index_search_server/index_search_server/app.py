import os
import logging
from flask import Flask, request
from index_search_server.index_search import HNSWIndexSearch
import json

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
index = HNSWIndexSearch()

@app.route('/best_matches', methods=['POST'])
def get_best_matches():
    data = request.get_json()
    if 'embedding' not in data or 'count' not in data:
        return json.dumps({})
    embedding, count = data['embedding'], data['count']
    idxs, distances = index.get_best_matches(embedding, count)
    response = {
        'result': [{'index': idx, 'distance': distance} for idx, distance in zip(idxs, distances)]
    }
    return json.dumps(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)