# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import numpy as np
# %%
url = "https://genius.com/The-beatles-help-lyrics"
#url =  "https://genius.com/The-beatles-norwegian-wood-this-bird-has-flown-lyrics"
req = requests.get(url) # Load the webpage
soup = BeautifulSoup(req.content, 'html.parser')
# %%
song_list = []
# %%
def get_dict(soup):
    song_dict = {}
    types = []
    texts = []

    title = soup.find("h1", class_="SongHeader__Title-sc-1b7aqpg-7 eJWiuG").get_text()
    album = soup.find("div", class_="HeaderTracklist__Album-sc-1qmk74v-3 hxXYDz").get_text()
    lyrics = soup.find("div", class_="SongPageGrid-sc-1vi6xda-0 DGVcp Lyrics__Root-sc-1ynbvzw-0 kkHBOZ")
    lyrics_str = str(lyrics.get_text(separator="<br/>")).strip() # Get text from lyrics section, keep line break
    lyrics_list = re.split("(\[.*?])", lyrics_str) # Split the section based on bracketss

    for sequence in lyrics_list: # For each sequence, save type and text
        name = re.findall("(\[.*?])", sequence) # Select names from brackets
        #remove_inline_brackets = re.sub(".(<br\/>)[^ ]", ". ", sequence)
        #print(remove_inline_brackets)
        #print("...")
        text = re.findall("^<br\/>.*", sequence) # Select text without line break tags

        if len(name) != 0:
            types.append(name)

        if len(text) != 0:
            texts.append(text)
    
    for type, text in zip(types, texts): # Put type and text into dict
        song_dict[type[0]] = text

    #print(types)
    #print(song_dict)
    #print(len(song_dict))
    print(song_dict.keys())
    i = 0
    while i < len(song_dict): # Creating the right format for names in dict
        new_key = str(re.findall("\[.*?]", types[i][0])[0]) # Text in first bracket
        #print(types[i])
        #print(i)
        old_key = str(types[i][0])
        try:
            song_dict[new_key] = song_dict[old_key]
            del song_dict[old_key]
        finally:
            i += 1
    #print(song_dict.keys())
    for key in song_dict: # Removing the brackets around the text
        song_dict[key] = song_dict[key][0]
    
    song_dict["title"] = title
    song_dict["album"] = album
    song_list.append(song_dict)
# %%
song_list = []
get_dict(soup)
#song_list
# %%
print(song_list[0].keys())
print(song_list)
# %%
song_list = []
url = "https://genius.com/The-beatles-help-lyrics"
req = requests.get(url) # Load the webpage
soup = BeautifulSoup(req.content, 'html.parser')
# %%
title = soup.find("h1", class_="SongHeader__Title-sc-1b7aqpg-7 jQiTNQ")
lyrics = soup.find("div", class_="Lyrics__Container-sc-1ynbvzw-6 krDVEH") # Get the whole lyrics section
# %%
lyrics_str = str(lyrics.get_text(separator="<br/>")).strip() # Get text from lyrics section, keep line break
lyrics_list = re.split("(\[.*?])", lyrics_str) # Split the section based on brackets
# %%
song_dict = {}
types = []
texts = []

for sequence in lyrics_list: # For each sequence, save type and text
    name = re.findall("(\[.*?])", sequence) # Select names from brackets
    remove_inline_brackets = re.sub(".(<br\/>)[^ ]", ". ", sequence)
    print(remove_inline_brackets)
    print("...")
    text = re.findall("^<br\/>.*", sequence) # Select text without line break tags

    if len(name) != 0:
        types.append(name)

    if len(text) != 0:
        texts.append(text)
# %%
for type, text in zip(types, texts): # Put type and text into dict
    song_dict[type[0]] = text
# %%
i = 0

while i < len(song_dict): # Creating the right format for names in dict
    new_key = str(re.findall("\[.*\d", types[i][0])[0] + "]") # Text up until number
    old_key = str(types[i][0])
    try:
        song_dict[new_key] = song_dict[old_key]
        del song_dict[old_key]
        i += 1
    except:
        i += 1
# %%
for key in song_dict: # Removing the brackets around the text
    song_dict[key] = song_dict[key][0]
# %%
for key in song_dict: # Removing line break tags from text
    #intext_b = re.findall(r"[a-zA-Z](<br\/>)", song_dict[key])
    #sub_intext_b = re.sub(r"[a-zA-Z](<br\/>)", ". ", song_dict[key])
    #sub_intext_b = re.sub(r"[a-zA-Z](<br\/>)", ". ", song_dict[key])
    #text_after = re.sub("<br\/>", " ", song_dict[key]).group().strip()
    #song_dict[key] = text_after    
    #print(song_dict[key])
    #print(sub_intext_b)
    #print(song_dict[key])
# %%
song_dict["[Verse 1]"]
# %%
