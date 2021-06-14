# %%
from numpy.lib.function_base import average
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
from textblob import TextBlob
from nltk.tokenize import RegexpTokenizer
from scipy import stats
# %%
# Loading new and old dataset
new_df = pd.read_csv("Cleanedtest.csv")
old_df = pd.read_csv("dataset.csv")
new_df["lyrics_cleaned"] = new_df["lyrics_cleaned"]
new_df = new_df.dropna(subset=['lyrics_cleaned'], axis=0,
               how='any')
# %%
# Checking character count before and after cleaning
joined_lyrics_old = " ".join(old_df["lyrics"])
joined_lyrics_cleaned = " ".join(new_df["lyrics_cleaned"])
print("Length of Lyrics Before Cleaning")
print(len(joined_lyrics_old))
print("Length of Lyrics After Cleaning")
print(len(joined_lyrics_cleaned))
# %%
# %%
# Function for removing special characters
def remove_special(inp):
    output = []
    for word in inp:
        if word.isalnum() is True:
            output.append(word)
    return output
# %%
# Function for keep non-stop words
stop_words = set(stopwords.words("english"))
def non_stop(inp):
    nonstop_list = []
    for w in inp:
        if w not in stop_words:
            nonstop_list.append(w)
    return nonstop_list
# %%
# Function for rejecting outliers
def reject_outliers(data, m=2): 
    return data[abs(data - mean(data)) < m * std(data)]
# %%
# Function for getting average length of section
tokenizer = RegexpTokenizer(r'\w+')
def get_avg_length(cat):
    len_list = []
    for row in cat:
        len_list.append(len(tokenizer.tokenize(row)))
    return len_list
# %%
# # Dropping nulls
intro = new_df["intro_cleaned"].dropna()
verse = new_df["verse_cleaned"].dropna()
bridge = new_df["bridge_cleaned"].dropna()
chorus = new_df["chorus_cleaned"].dropna()
outro = new_df["outro_cleaned"].dropna()
lyrics_list = new_df["lyrics_cleaned"].dropna()
cat_full = [intro, verse, bridge, chorus, outro, lyrics_list]
# %%
# Tokenizing sections
intro_tokens = remove_special(word_tokenize(" ".join(intro)))
verse_tokens = remove_special(word_tokenize(" ".join(verse)))
bridge_tokens = remove_special(word_tokenize(" ".join(bridge)))
chorus_tokens = remove_special(word_tokenize(" ".join(chorus)))
outro_tokens = remove_special(word_tokenize(" ".join(outro)))
lyrics_list_tokens = remove_special(word_tokenize(" ".join(lyrics_list)))
# %%
# Keeping non-stop words
intro_nonstop = non_stop(intro_tokens)
verse_nonstop = non_stop(verse_tokens)
bridge_nonstop = non_stop(bridge_tokens)
chorus_nonstop = non_stop(chorus_tokens)
outro_nonstop = non_stop(outro_tokens)
lyrics_nonstop = non_stop(lyrics_list_tokens)
categories = [intro_nonstop, verse_nonstop, bridge_nonstop, chorus_nonstop, outro_nonstop, lyrics_nonstop]
# %%
# Extracting full lyrics
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
lyrics = extract_text(new_df["lyrics_cleaned"])
# %%
# Checking stem and lemma unique count of full lyrics set
port_stemmer_lyrics_cleaned = []
lanc_stemmer_lyrics_cleaned = []
#wordnet_lemma_lyrics_cleaned = []
for w in lyrics:
    port_stemmer_lyrics_cleaned.append(PorterStemmer().stem(w))
    lanc_stemmer_lyrics_cleaned.append(LancasterStemmer().stem(w))
    #wordnet_lemma_lyrics_cleaned.append(WordNetLemmatizer().lemmatize(w))
port_unique = set(port_stemmer_lyrics_cleaned)
lanc_unique = set(lanc_stemmer_lyrics_cleaned)
#lem_unique = set(wordnet_lemma_lyrics_cleaned)

print("Porter Unique:", len(port_unique))
print("Lancaster Unique:", len(lanc_unique))
#print("Lemmatizer Unique:", len(lanc_unique))
print("Total Unique:", len(set(lyrics)))
# %%
# Getting 10 most common words for each section
for cat in categories:
    print(Counter(cat).most_common(10))
# %%
# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
for cat in cat_full:
    print(sia.polarity_scores(" ".join(cat)))
# Sentiment Analysis - Subjectivity
for cat in cat_full:
    text = TextBlob(" ".join(cat))
    print(text.sentiment)
# %%
# Box-Plot for each section word count and third quantile (NB: Lyrics is concatenated)
plt.figure(figsize=(10,5))
plt.suptitle("Section Word Count")
plt.figtext(1, 0.001, "Removed outliers 2 standard deviations away from the mean", wrap=True, horizontalalignment='right', fontsize=8)
plt.subplot(2,3,1)
plt.boxplot(reject_outliers(np.array(get_avg_length(intro))))
plt.title("Intro")
print(quantile(reject_outliers(np.array(get_avg_length(intro))), 0.75))

plt.subplot(2,3,2)
plt.boxplot(reject_outliers(np.array(get_avg_length(verse))))
plt.title("Verse")
print(quantile(reject_outliers(np.array(get_avg_length(verse))), 0.75))

plt.subplot(2,3,3)
plt.boxplot(reject_outliers(np.array(get_avg_length(bridge))))
plt.title("Bridge")
print(quantile(reject_outliers(np.array(get_avg_length(bridge))), 0.75))

plt.subplot(2,3,4)
plt.boxplot(reject_outliers(np.array(get_avg_length(chorus))))
plt.title("Chorus")
print(quantile(reject_outliers(np.array(get_avg_length(chorus))), 0.75))

plt.subplot(2,3,5)
plt.boxplot(reject_outliers(np.array(get_avg_length(outro))))
plt.title("Outro")
print(quantile(reject_outliers(np.array(get_avg_length(outro))), 0.75))

plt.subplot(2,3,6)
plt.boxplot(reject_outliers(np.array(get_avg_length(lyrics_list))))
plt.title("Full Lyrics")
plt.tight_layout()
plt.savefig("section_word_count")
plt.show()
# %%
# Average Line Length
df_line = pd.read_csv("cleaned_dataset.csv")
df_line = df_line.lyrics_semi_cleaned

line_length = []
for row in df_line:
    line_length.append(row.split("\n"))

sent_length = []
for row in line_length:
    for sent in row:
        sent_length.append(len(sent.split(" ")))

print(average(reject_outliers(np.array(sent_length))))
print(average(sent_length))
# %%
