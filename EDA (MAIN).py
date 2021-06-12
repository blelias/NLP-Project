# %%
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from numpy import mean
from numpy import quantile
from numpy import std
import re
nltk.download("vader_lexicon")
# %%
old_df = pd.read_csv("dataset.csv")
new_df = pd.read_csv("Cleaned.csv")
new_df = new_df[0:310]
# %%
new_df["lyrics_cleaned"] = new_df["lyrics_cleaned"].dropna()
# %%
joined_lyrics_old = " ".join(old_df["lyrics"])
joined_lyrics_cleaned = " ".join(new_df["lyrics_cleaned"])
# %%
########## Characters before and after cleaning
print("Length of Lyrics Before Cleaning")
print(len(joined_lyrics_old))
print("Length of Lyrics After Cleaning")
print(len(joined_lyrics_cleaned))
# %%
########## Extracting Text
def extract_text(lyrics):
    global corpus
    lst = []
    for i in range(len(lyrics)):
        lst.append(str(lyrics[i]))
    data = ''.join(lst)
    
    corpus = []
    corpus += [w for w in data.split(' ') if w.strip() != '' or w == '\n']
    lyrics = ' '.join(corpus)
    
    return lyrics
# %%
lyrics = extract_text(new_df["lyrics_cleaned"])
# %%
########## Creating Vocabulary
def vocab(lyrics, char = False): 
    if char == True:
        vocab = sorted(set(lyrics))
        print(f'{len(vocab)} unique characters in vocabulary')
    else:
        lyrics = [w for w in lyrics.split(' ')]
        vocab = sorted(set(lyrics))
        print(f'{len(vocab)} unique words in vocabulary')
    return lyrics, vocab

lyrics, vocab = vocab(lyrics)

print(type(lyrics)); print(type(vocab))
# %% # Dropping nulls
intro = new_df["intro_cleaned"].dropna()
verse = new_df["verse_cleaned"].dropna()
bridge = new_df["bridge_cleaned"].dropna()
chorus = new_df["chorus_cleaned"].dropna()
outro = new_df["outro_cleaned"].dropna()
lyrics_list = new_df["lyrics_cleaned"].dropna()
cat_full = [intro, verse, bridge, chorus, outro, lyrics_list]
# %%
# Removing Special Characters
def remove_special(inp):
    output = []
    for word in inp:
        if word.isalnum() is True:
            output.append(word)
    return output
# %%
intro_tokens = remove_special(word_tokenize(" ".join(intro)))
verse_tokens = remove_special(word_tokenize(" ".join(verse)))
bridge_tokens = remove_special(word_tokenize(" ".join(bridge)))
chorus_tokens = remove_special(word_tokenize(" ".join(chorus)))
outro_tokens = remove_special(word_tokenize(" ".join(outro)))
lyrics_list_tokens = remove_special(word_tokenize(" ".join(lyrics_list)))
# %%
######### Checking stem and lemma unique count of full lyrics set
port_stemmer_lyrics_cleaned = []
lanc_stemmer_lyrics_cleaned = []
wordnet_lemma_lyrics_cleaned = []
for w in lyrics:
    port_stemmer_lyrics_cleaned.append(PorterStemmer().stem(w))
    lanc_stemmer_lyrics_cleaned.append(LancasterStemmer().stem(w))
    wordnet_lemma_lyrics_cleaned.append(WordNetLemmatizer().lemmatize(w))
port_unique = set(port_stemmer_lyrics_cleaned)
lanc_unique = set(lanc_stemmer_lyrics_cleaned)
lem_unique = set(wordnet_lemma_lyrics_cleaned)
# %%
print("Porter Unique:", len(port_unique))
print("Lancaster Unique:", len(lanc_unique))
print("Lemmatizer Unique:", len(lanc_unique))
print("Total Unique:", len(set(lyrics)))
# %%
########## Checking most popular non-stop-words 
stop_words = set(stopwords.words("english"))
# %%
#%%
########## Function for keep non-stop words
def non_stop(inp):
    nonstop_list = []
    for w in inp:
        if w not in stop_words:
            nonstop_list.append(w)
    return nonstop_list
# %%
########## Keeping non-stop words
intro_nonstop = non_stop(intro_tokens)
verse_nonstop = non_stop(verse_tokens)
bridge_nonstop = non_stop(bridge_tokens)
chorus_nonstop = non_stop(chorus_tokens)
outro_nonstop = non_stop(outro_tokens)
lyrics_nonstop = non_stop(lyrics_list_tokens)
categories = [intro_nonstop, verse_nonstop, bridge_nonstop, chorus_nonstop, outro_nonstop, lyrics_nonstop]
# %%
len(set(verse_nonstop))
# %%
sorted(verse_nonstop)
# %%
########## Getting most common words for each category
for cat in categories:
    Counter(cat).most_common(10)
# %%
########## Frequency Distribution
fd = nltk.FreqDist(verse_nonstop)
lower_fd = nltk.FreqDist([w.lower() for w in fd])
# %%
########## Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(" ".join(lyrics))
# %%
# Intro
#{'neg': 0.064, 'neu': 0.65, 'pos': 0.286, 'compound': 0.9996}
# Verse
#{'neg': 0.086, 'neu': 0.71, 'pos': 0.204, 'compound': 1.0}
# Bridge
# {'neg': 0.085, 'neu': 0.685, 'pos': 0.23, 'compound': 1.0}
# Chorus
# 'neg': 0.087, 'neu': 0.703, 'pos': 0.209, 'compound': 1.0
# Outro
# {'neg': 0.053, 'neu': 0.672, 'pos': 0.275, 'compound': 1.0}
# Lyrics
# {'neg': 0.081, 'neu': 0.711, 'pos': 0.209, 'compound': 1.0}
# %% 
# Rejecting Outliers
def reject_outliers(data, m=1): 
    return data[abs(data - mean(data)) < m * std(data)]

#length_arr = array(length_list)
#length_list = list(reject_outliers(length_arr))
# %%
########## Box-Plot of length of each section
def get_boxplot(inp):
    my_list = []
    for x in inp:
        my_list.append(len(word_tokenize(x)))
    return list(reject_outliers(array(my_list))), my_list
# %%
########## Box-Plot for each section
for cat in cat_full:
    plt.figure()
    plt.boxplot(get_boxplot(cat)[0])
    plt.show
# %%
########## Getting interquartile range for length of each section (but this is total, e.g. with all verses from a song)
intro_nonstop
for cat in cat_full:
    q1 = quantile(get_boxplot(cat)[1], 0.25)
    print(get_boxplot(cat)[1])
    q3 = quantile(get_boxplot(cat)[1], 0.75)
    #avg_len = float(round(avg_len))
    print("Q1 quantile: " + str(q1), "\nQ3 quantile: " + str(q3))
# %%
len(" when i find myself in times of trouble, mother mary comes to me speaking words of wisdom, let it be and in my hour of darkness, she is standing right in front of me speaking words of wisdom, let it be', and when the brokenhearted people living in the world agree there will be an answer, let it be for though they may be parted, there is still a chance that they will see there will be an answer, let it be', and when the night is cloudy, there is still a light that shines on me shine on 'til tomorrow, let it be i wake up to the sound of music, mother mary comes to me speaking words of wisdom, let it be'")
# %%
########## Finding interquartile range for line length ( This is from the other dataset )
#%%
df = pd.read_csv("cleaned_dataset.csv")
def get_data(col):
    return ".".join(list(df[col].dropna())) # dropna is probably unnecessary
data_lyrics_break = get_data("lyrics_semi_clean") # This series contains the lyrics w/ line break tags
#%%
# Finding numerical measures for length of text lines
lyrics_joined_break = "".join(data_lyrics_break)
lyrics_joined_break = re.sub("[\n\n]{2,}", "\n", lyrics_joined_break)

sents_break = lyrics_joined_break.split('\n')
avg_len = round(sum(len(x.split(" ")) for x in sents_break) / len(sents_break), 2) # Average length of line (With outliers)
length_list = [len(x.split(" ")) for x in sents_break]
#print("Average line length: " + str(avg_len))
#plt.boxplot(length_list)
#plt.show()
# %%
def reject_outliers(data, m=2): # Rejecting Outliers
    return data[abs(data - mean(data)) < m * std(data)]
# %%
length_arr = array(length_list)
length_list = list(reject_outliers(length_arr))
avg_len = round(sum(length_list) / len(length_list), 2) # Average length of line (without outliers)

print("After removing outliers: " + str(avg_len))
plt.boxplot(length_list)
plt.title("Average Line Length Across Sections (Outliers Removed")
plt.show()
plt.hist(length_list)
plt.title("Average Line Length Across Sections (Outliers Removed")
plt.show()
# %%
categories = [data_lyrics_break]
#def line_lengths(categories):
lyrics_joined_break_verse = "".join(data_lyrics_break)
lyrics_joined_break_verse = re.sub("[\n\n]{2,}", "\n", lyrics_joined_break)
sents_break_verse = lyrics_joined_break.split('\n')
length_arr = array(length_list)
length_list = list(reject_outliers(length_arr))
avg_len = round(sum(length_list) / len(length_list), 2) # Average length of line (without outliers)
# %%
len(sents_break_verse)
sents_break_verse[5]
# %%
# EDA Plotting
plt.boxplot(length_list)
plt.title("Line length - All Categories")
plt.show()
plt.hist(length_list)
plt.title("Line length - All Categories")
plt.show()
# %%
q1 = quantile(length_arr, 0.25)
q3 = quantile(length_arr, 0.75)
avg_len = float(round(avg_len))
print("Q1 quantile: " + str(q1), "\nQ3 quantile: " + str(q3) +
"\nRounded average length: " + str(avg_len))
# %%
