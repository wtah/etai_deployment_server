import requests
url = 'http://127.0.0.1:8000/'
files = {'sample': open('sample_data/image/hermes-rivera-qbf59TU077Q-unsplash.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())