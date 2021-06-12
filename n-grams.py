# %%
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from nltk import trigrams
import numpy as np
from numpy import mean, std, quantile
from numpy import array
# %%
df = pd.read_csv("cleaned_dataset.csv")
# %%
def get_data(col):
    return ".".join(list(df[col].dropna())) # dropna is probably unnecessary

data_intro = get_data("intro_cleaned")
data_verse = get_data("verse_cleaned")
data_bridge = get_data("bridge_cleaned")
data_chorus = get_data("chorus_cleaned")
data_outro = get_data("outro_cleaned")
data_lyrics = get_data("lyrics_cleaned")
data_lyrics_break = get_data("lyrics_semi_clean") # This series contains the lyrics w/ line break tags
data_title = get_data("title")
# %%
# Creating trigram model to generate closing word of each line
text = [None, None]
model_sample = defaultdict(lambda: defaultdict(lambda: 0))
sents = sent_tokenize(data_lyrics)
corpus_sents = []

for sent in sents:
    corpus_sents.append(word_tokenize(sent))

for sentence in corpus_sents:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model_sample[(w1, w2)][w3] += 1

ending_words = model_sample[tuple(text[-2:])].keys()
#ending_words = [x for x in ending_words if not any(c.isdigit() for c in x)] # Removing numbers, but this is already done in the cleaning
# %%
# Finding numerical measures for length of text lines
lyrics_joined_break = "".join(data_lyrics_break)
lyrics_joined_break = re.sub("[\n\n]{2,}", "\n", lyrics_joined_break)

sents_break = lyrics_joined_break.split('\n')
avg_len = round(sum(len(x.split(" ")) for x in sents_break) / len(sents_break), 2) # Average length of line (With outliers)
length_list = [len(x.split(" ")) for x in sents_break]
print("Average line length: " + str(avg_len))
plt.boxplot(length_list)
plt.show()
# %%
def reject_outliers(data, m=2): # Rejecting Outliers
    return data[abs(data - mean(data)) < m * std(data)]
# %%
length_arr = array(length_list)
length_list = list(reject_outliers(length_arr))
avg_len = round(sum(length_list) / len(length_list), 2) # Average length of line (without outliers)

print("After removing outliers: " + str(avg_len))
plt.boxplot(length_list)
plt.show()
plt.hist(length_list)
plt.show()
# %%
# Line lengths accross categories
lyrics_joined_break_verse = "".join(data_lyrics_break)
lyrics_joined_break_verse = re.sub("[\n\n]{2,}", "\n", lyrics_joined_break)
sents_break_verse = lyrics_joined_break.split('\n')

df["verse"][0]
# Topic Modellin
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
# Creating unigram model for generating the two first words of each line
def unigram_get_words(data, length):
    tokenizer = RegexpTokenizer(r'\w+')
    text = data
    tokens = tokenizer.tokenize(text)
    counts = Counter(tokens)
    total_count = len(tokens)

    for word in counts:
        counts[word] /= float(total_count)

    text = []
    for i in range(length):
        r = random.random()
        accumulator = .0
        for word, freq in counts.items():
            accumulator += freq
            if accumulator >= r:
                text.append(word)
                break
    return " ".join(text)
# %%
def create_song(format, specialized, i_len=5, b_len=5, c_len=10, v_len=20, o_len=5):
    title = unigram_get_words(data_title, 4)
    song = []
    #song.append("SONG NAME: " + title.upper() + "\n")
    if specialized is True:
        intro = "[INTRO]\n" + get_sequence(model_intro[0], model_intro[1], model_intro[2]-1, unigram_get_words(data_lyrics, 2), i_len) # Parameters: model, tokenizer, max_length
        bridge = "[BRIDGE]\n" + get_sequence(model_bridge[0], model_bridge[1], model_bridge[2]-1, unigram_get_words(data_lyrics, 2), b_len)
        chorus = "[CHORUS]\n" + get_sequence(model_chorus[0], model_chorus[1], model_chorus[2]-1, unigram_get_words(data_lyrics, 2), c_len)
        outro = "[OUTRO]\n" + get_sequence(model_outro[0], model_outro[1], model_outro[2]-1, unigram_get_words(data_lyrics, 2), o_len)
        pass
    else:
        intro = "[INTRO]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), i_len)
        bridge = "[BRIDGE]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), b_len)
        chorus = "[CHORUS]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), c_len)
        outro = "[OUTRO]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), o_len)
    for tag in format:
        if tag == "I":
            song.append(intro)
        elif tag == "V":
            if specialized is True:
                verse = "[VERSE]\n" + get_sequence(model_verse[0], model_verse[1], model_verse[2]-1, unigram_get_words(data_lyrics, 2), v_len)
            else:
                verse = "[VERSE]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), v_len)
            song.append(verse)
        elif tag == "B":
            song.append(bridge)
            pass
        elif tag == "C":
            song.append(chorus)
            pass
        elif tag == "O":
            song.append(outro)
            pass
    return song
# %%
def print_song(song):
    for element in song:
        print(element)
# %%
song_specialized = create_song("IVCVCBO", specialized=True)
# %%
########## N-Gram, take 2