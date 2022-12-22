import requests
url = 'http://127.0.0.1:8000/'

def send_request(text):
    data = {'sample': text}
    response = requests.post(url, data=data)
    print(response.json())

sentences = ['I love this article', 'This article has alot of words.', 'This article is bad, very bad. Bad bad article.']

for sent in sentences:
    send_request(text=sent)