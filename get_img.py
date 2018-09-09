import requests
from requests.compat import urljoin
from bs4 import BeautifulSoup

def save_img(url):
    images = []
    soup = BeautifulSoup(requests.get(url).content,'lxml') 

    for link in soup.find_all("img"):
        if link.get("src").endswith(".jpg"): 
            images.append(urljoin(url, link.get("src"))) 
        elif link.get("src").endswith(".png"):
            images.append(urljoin(url, link.get("src")))

    del images[0]

    for i in images:
        re = requests.get(i)
        with open('./img/'+ i.split('/')[-1], 'wb') as f:
            f.write(re.content) 
    print('done')
    
save_img(input('URL：'))
