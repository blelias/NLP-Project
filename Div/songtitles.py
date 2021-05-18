# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
# %%
url= "https://open.spotify.com/search/the%20beatles/tracks"
req = requests.get(url) # Load the webpage
print(req.status_code)
soup = BeautifulSoup(req.content, 'html.parser')
# %%
text = soup.find("div", class_="_7effa9d9b3900e9698aa6e0423a1e841-scss _98a17d59ea3df3c60b9699a6afe43816-scss").get_text()
# %%
soup
# %%
