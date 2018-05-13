import requests

url = 'http://localhost:8081/best_matches'
payload = {
    'embedding': [x for x in range(128)],
    'count':2
}
r = requests.post(url, json=payload)
print(r.status_code)
print(r.json())
