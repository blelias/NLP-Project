# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import numpy as np
import time
import os

# %%
def request_artist_info(artist_name, page):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + 'rH4JGMsJRWH3jELBQ9opqAJX0QROiIMYOXXTYIR7DoNwb0t9DIgowIibgRV7t3Md'}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    return response
# %%
def request_song_url(artist_name, song_cap):
    page = 1
    songs = []
    
    while True:
        response = request_artist_info(artist_name, page)
        json = response.json()
        
        song_info = []
        for hit in json['response']['hits']:
            if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
                song_info.append(hit)

        for song in song_info:
            if (len(songs) < song_cap):
                url = song['result']['url']
                songs.append(url)
            
        if (len(songs) == song_cap):
            break
        else:
            page += 1
    return songs

urls = request_song_url('The Beatles', 311)
# %%
def retrive_info(url):
    lyrics = None
    timeout = time.time() + 15 #15 seconds
    while lyrics == None and time.time() < timeout:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        lyrics = soup.find("div", class_="lyrics")

    #two ways of getting lyris from geius
    lyrics1 = soup.find("div", class_="lyrics")

    #check for lyrics.
    if lyrics1:
        lyrics = lyrics1.get_text()
    elif lyrics1 == None:
        lyrics = None
    
    #Intro, Verse, Chorus. Outro
    lyrics1 = lyrics.split('\n\n',15)
    lst1 = []
    lst2 = []
    lst3 = []
    lst4 = []
    lst5 = []
    for i in range(len(lyrics1)):
        if '[Intro' in lyrics1[i]:
            lst1.append(lyrics1[i])
        elif '[Verse' in lyrics1[i]:
            lst2.append(lyrics1[i])
        elif '[Chorus' in lyrics1[i]:
            lst3.append(lyrics1[i])
        elif '[Bridge' in lyrics1[i]:
            lst4.append(lyrics1[i])
        elif '[Outro' in lyrics1[i]:
            lst5.append(lyrics1[i])
            

    # title
    title = soup.find('h1',class_="header_with_cover_art-primary_info-title")
    if title != None:
        title = title.text.strip()
    else: 
        title = None

    #date
    try: 
        date = soup.findAll('span',class_="metadata_unit-info metadata_unit-info--text_only")
        date = date[1].text.strip()
    except IndexError:
        try: 
            date = soup.findAll('span',class_="metadata_unit-info metadata_unit-info--text_only")
            date = date[0].text.strip()
        except:
            date  = None 

    #album
    try:
        album = soup.findAll('span', class_="metadata_unit-info")
        album = album[1].text.strip()
    except: 
        album = None

    return title,date,album,lyrics,lst1,lst2,lst3,lst4,lst5
# %%
def make_df(urls):
    df = pd.DataFrame(columns = ['title', 'date', 'album', 'lyrics', 'intro', 'verse', 'chorus','bridge', 'outro'])
    for i, url in enumerate(urls):
        title,date,album,lyrics,lst1,lst2,lst3,lst4,lst5 = retrive_info(url)
        df.loc[i,'title'] = title
        df.loc[i,'date'] = date
        df.loc[i,'album'] = album
        df.loc[i,'lyrics'] = lyrics
        df.at[i,'intro'] = lst1
        df.at[i,'verse'] = lst2
        df.at[i,'chorus'] = lst3
        df.at[i,'bridge'] = lst4
        df.at[i,'outro'] = lst5
        df
        print(url)
    return df
# %%
df = make_df(urls)
df.to_csv("dataset.csv")
# %%
df = pd.read_csv("dataset.csv")
# %%
def clean(s):

    # Format words and remove unwanted characters
    s = re.sub(r'[\(\[].*?[\)\]]', '', s)
    s = os.linesep.join([i for i in s.splitlines() if i])
    s = s.replace("\\n", " ")
    s = re.sub(r"[^'.,a-zA-Z0-9 \.-]+", '', s)
    s = re.sub(r"\s'\s", " ", s)
    s = s.lower()
    return s
# %%
df['intro_cleaned'] = list(map(clean, df.intro))
df['verse_cleaned'] = list(map(clean, df.verse))
df['bridge_cleaned'] = list(map(clean, df.bridge))
df['chorus_cleaned'] = list(map(clean, df.chorus))
df['outro_cleaned'] = list(map(clean, df.outro))
# %%
df.to_csv("cleaned_dataset.csv")
# %%
