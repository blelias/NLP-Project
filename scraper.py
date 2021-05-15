# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
# %%
song_list = []
def get_song(soup):
    song_dict = {}

    # Get the text values from website
    #title = soup.find("h1", class_="SongHeader__Title-sc-1b7aqpg-7 eJWiuG").get_text()
    #album = soup.find("div", class_="HeaderTracklist__Album-sc-1qmk74v-3 hxXYDz").get_text()
    #lyrics = soup.find("div", class_="SongPageGrid-sc-1vi6xda-0 DGVcp Lyrics__Root-sc-1ynbvzw-0 kkHBOZ").get_text(separator="<br/>")
    title = soup.find("h1").get_text()
    print(title)
    album = soup.find("div", class_=re.compile(r".*?_Album-.*?")).get_text()
    print(album)
    lyrics = soup.find("div", class_=re.compile(r".*?Lyrics__Container.*?")).get_text(seperator="<br/>")
    print(lyrics)
    
    # HOLDER PÅ MED: få soup.find til å funke (gjør slik at den targeter hele klassenavnet med regex)

    # Split text based on bracket location
    split_text = re.split(r"\[.*?]", lyrics)
    # Save content of brackets as type
    split_type = re.findall(r"\[.*?]", lyrics)
    # Remove first type, which is empty
    split_text.pop(0)

    i = 2
    for type, text in zip(split_type, split_text):
        if type in song_dict: # If the type is already a key, add counter
            type += str(i)
            i += 1
        type = re.sub(r"\s]", "]", type)
        song_dict[type] = text
    song_dict["title"] = title
    song_dict["album"] = album
    song_list.append(song_dict)
# %%
def load_url(url_list):
    for url in url_list:
        req = requests.get(url) # Load the webpage
        soup = BeautifulSoup(req.content, 'html.parser')
        print(req.status_code)
        get_song(soup)
# %%
url_list = ["https://genius.com/John-lennon-imagine-lyrics", "https://genius.com/The-beatles-help-lyrics", "https://genius.com/The-beatles-yellow-submarine-lyrics"]
#url_list = ["https://genius.com/The-beatles-yellow-submarine-lyrics"]
song_list = []
load_url(url_list)
# %%
song_list
# %%
