# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import numpy as np
# %%
url = "https://genius.com/The-beatles-help-lyrics"
req = requests.get(url) # Load the webpage
soup = BeautifulSoup(req.content, 'html.parser')
song_list = []
# %%
def get_song(soup):
    song_dict = {}
    title = soup.find("h1", class_="SongHeader__Title-sc-1b7aqpg-7 eJWiuG").get_text()
    album = soup.find("div", class_="HeaderTracklist__Album-sc-1qmk74v-3 hxXYDz").get_text()
    lyrics = soup.find("div", class_="SongPageGrid-sc-1vi6xda-0 DGVcp Lyrics__Root-sc-1ynbvzw-0 kkHBOZ").get_text(separator="<br/>")
# %%
get_song(soup)
# %%
lyrics = soup.find("div", class_="SongPageGrid-sc-1vi6xda-0 DGVcp Lyrics__Root-sc-1ynbvzw-0 kkHBOZ").get_text(separator="<br/>")
# %%
split_text = re.split(r"\[.*?]", lyrics)
split_type = re.findall(r"\[.*?]", lyrics)
split_text.pop(0
# %%
for type, text in zip(split_type, split_text):
    song_dict[type] = text
# %%
song_dict = {}
# %%
song_dict
# %%
