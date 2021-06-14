# %%
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from collections import Counter, defaultdict
import random
from nltk import data, trigrams, sent_tokenize, word_tokenize
from nltk.corpus import reuters
# %%
df = pd.read_csv("cleaned_dataset.csv")

def get_data(col):
    return ".".join(list(df[col].dropna())) # dropna is probably unnecessary
# %%
data_intro = get_data("intro_cleaned")
data_verse = get_data("verse_cleaned")
data_bridge = get_data("bridge_cleaned")
data_chorus = get_data("chorus_cleaned")
data_outro = get_data("outro_cleaned")
data_lyrics = get_data("lyrics_cleaned")
data_lyrics_break = get_data("lyrics_semi_clean") # This series contains the lyrics w/ line break tags
data_title = get_data("title")
# %%
# Rules based on quantiles from EDA
# Max word length of each section
i_len=18 # fourth quantile
b_len=136 # fourth quantile, Gets divided by number of verses
c_len=54 # fourth quantile
v_len=94 # fourth quantile
o_len=22 # fourth quantile

# Line lengths of each section (This is currently in the EDA)
line_length_i = 5
line_length_b = 5
line_length_c = 5
line_length_v = 5
line_length_o = 5
# %%
# Unigram Model
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
tokenizer = RegexpTokenizer(r'\w+')
# %%
# Create Trigram Model
def create_trigram(data):
    sample_text = data
    sents = sent_tokenize(sample_text)
    corpus_sents = []

    for sent in sents:
        #corpus_sents.append(word_tokenize(sent))
        corpus_sents.append(tokenizer.tokenize(sent))
    model_sample = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in corpus_sents:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model_sample[(w1, w2)][w3] += 1

    for w1_w2 in model_sample:
        total_count = float(sum(model_sample[w1_w2].values()))
        if total_count == 0:
            print(w1_w2, ':', total_count, ':', model_sample[w1_w2])

        for w3 in model_sample[w1_w2]:
            if total_count == 0:
                model_sample[w1_w2][w3] = 0
            else:
                model_sample[w1_w2][w3] /= total_count
    return model_sample
# %%
# Use Trigram Model
def use_trigram(model, n_words, line_length):
    text = [None, None]
    sentence_finished = False
    x = 0
    i = 0

    while not sentence_finished:
        r = random.random()
        accumulator = .0
        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            if accumulator >= r:
                text.append(word)
                x += 1
                break
        if text[-2:] == [None, None] and x >= n_words*0.50:
            sentence_finished = True
        elif x >= n_words:
            sentence_finished = True
        #elif len(text) > 10:
            #sentence_finished = True
    text = [t for t in text if t]
    for idx, word in enumerate(text):
        i += 1
        if i >= line_length:
            text[idx] += "\n"
            i = 0
    out_text = ' '.join(text).lower()
    return out_text
# %%
def create_song(format, data, specialized=True, i_len=i_len, b_len=b_len, c_len=c_len, v_len=v_len, o_len=o_len):
    title = unigram_get_words(data_title, 4)
    song = []
    #song.append("SONG NAME: " + title.upper() + "\n")
    if specialized is True:
        intro = "[INTRO]\n" + use_trigram(create_trigram(data_intro),i_len, line_length_i) # Parameters: model, max_length
        bridge = "[BRIDGE]\n" + use_trigram(create_trigram(data_bridge),b_len, line_length_b)
        chorus = "[CHORUS]]\n" + use_trigram(create_trigram(data_chorus),c_len, line_length_c)
        outro = "[OUTRO]\n" + use_trigram(create_trigram(data_outro),o_len, line_length_o)
    else:
        model = create_trigram(data_lyrics)
        intro = "[INTRO]\n" + use_trigram(model,i_len, line_length_i) # Parameters: model, max_length
        bridge = "[BRIDGE]\n" + use_trigram(model,b_len, line_length_b)
        chorus = "[CHORUS]]\n" + use_trigram(model,c_len, line_length_c)
        outro = "[OUTRO]\n" + use_trigram(model,o_len, line_length_o)
    for tag in format:
        if tag == "I":
            song.append(intro)
        elif tag == "V":
            if specialized is True:
                verse = "[VERSE]\n" + use_trigram(create_trigram(data_verse),v_len/format.count("V"), line_length_v)
            else:
                verse = "[VERSE]\n" + use_trigram(model,v_len/format.count("V"), line_length_v)
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
    return song, title

# %%
def print_song(song):
    print("Title: " + str(song[1]).upper() + "\n")
    for element in song[0]:
        print(element + "\n")
# %%
print_song(create_song("IVCVCO", data_intro, specialized=True))
# %%

# %%
