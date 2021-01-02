import requests
import json

url = 'http://127.0.0.1:8080/predict_lightgbm'

data = [{'customer':'C2079384186','age':'2','gender':'F','merchant':'M1294758098','category':'es_leisure','amount':'10'}]
#data = ["C1949984685",0,"F","M1294758098","es_leisure",200]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)