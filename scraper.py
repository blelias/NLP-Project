# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
# %%
df_features = pd.read_csv("song_features_dataset.csv")
# %%
song_list = []
def get_song(soup, title):
    song_dict = {}
    # If the title does not exist in song_list
    if not any(d['title'] == title for d in song_list): # Fix this
        try:
            print(title)
            #
            # album = soup.find("div", class_=re.compile(r'_Album-.*?"'[0:-1])).get_text()
            lyrics = soup.find("div", class_=re.compile(r'"Lyrics__Container.*?"'[1:-1])).get_text(separator="<br/>") # .*?Lyrics__Container.*?
            print("Success")
        except:
            print("Error")
            return
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
        #song_dict["album"] = album
        song_list.append(song_dict)
    else:
        print("Error: This iteration should not have happend")
# %%
def load_url(n_songs): # Call get_song function
    url_list = []
    for i in range(n_songs): # Choose number of songs in URL-list
        song_title = df_features.iloc[i][0]
        song_title = standard_title(song_title)       
        url_list.append(create_url(song_title))
    #print(url_list)
    for i in range(20): # Choose number of iterations
        for url in url_list:
            # If the title from the URL does not exisit in song_list, continue
            if not any(standard_title(d['title']) == standard_title(re.findall(r"beatles-(.*)-lyrics", url)[0]) for d in song_list):
                req = requests.get(url) # Load the webpage
                soup = BeautifulSoup(req.content, 'html.parser')
                if req.status_code == 404:
                    url_list.remove(url) # url_list.remove(create_url(title))
                    print("Could not find " + url)
                    break
                print(req.status_code)
                title = soup.find("h1").text.strip()
                get_song(soup, title)

                # If successful, remove url from url-list
                if any(standard_title(d['title']) == standard_title(re.findall(r"beatles-(.*)-lyrics", url)[0]) for d in song_list): 
                    url_list.remove(url) # url_list.remove(create_url(title))
                    print("Removed " + url)
# %%
def standard_title(title):
    title = title.replace(' ', '-')
    title = title.replace("'", '')
    title = title.replace("’", "")
    title = title.replace("(", "")
    title = title.replace(")", "")
    return title
def create_url(title):
    url = "https://genius.com/The-beatles-" + standard_title(title) + "-lyrics"
    return url
# %%
#song_list = []
load_url(100) # Choose number of songs
to_csv(song_list)
# %%
# Function for adding song list to pandas dataframe
def to_csv(song_list):
    df = pd.DataFrame(song_list)
    df.to_csv("lyrics_dataset.csv")
# %%
len(song_list)
# %%
