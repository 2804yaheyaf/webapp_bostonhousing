import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'rm':6.5, 'tax':0.4, 'ptratio':0.6, 'lstat':0.3, 'indus':12})

print(r.json())