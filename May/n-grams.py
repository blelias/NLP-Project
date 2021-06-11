# %%
from itertools import filterfalse
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from collections import defaultdict
from nltk import trigrams
import random
# %%
df = pd.read_csv("cleaned_dataset.csv")
data = df.lyrics_cleaned.dropna()
data = " ".join(data)
# %%
q3 = 8
# %%
def create_trigram(data):
    model = defaultdict(lambda: defaultdict(lambda: 0)) # Model that will be the output
    sents = sent_tokenize(data) # Use sentence tokenizer on the input data
    corpus_sents = []

    for sent in sents:
        corpus_sents.append(word_tokenize(sent)) # Append the word tokenized sentences

    for sent in corpus_sents:
        for word1, word2, word3 in trigrams(sent, pad_right=True, pad_left=True):
            model[(word1, word2)][word3] += 1 # For each trigram in each sentence add a key word1, word2 to the dictionary with the third word as value

    for word1_word2 in model:
        tot_count = float(sum(model[word1_word2].values())) # Total count is the sum of the values for each key in the model

        for word3 in model[word1_word2]: # For each value for each key
            if tot_count == 0: # If the count is zero, set value to zero
                model[word1_word2][word3] = 0 
            else:
                model[word1_word2][word3] /= tot_count # Else set the value to the proportion of the total count
    return model
# %%
model_fullset = create_trigram(data)
# %%
def generate_text(model, length):
    module = []
    for i in range(length):
        text = [None, None]
        sent_finished = False

        while not sent_finished:
            r = random.random()
            accumulator = 0.0
            for word in model[tuple(text[-2:])].keys():
                accumulator += model[tuple(text[-2:])][word]
                if accumulator >= r:
                    text.append(word)
                    break
            if text[-2:] == [None, None] or len(text) > q3:
                sent_finished = True
        module.append(' '.join([t for t in text if t]))
    return module
# %%
def create_song(format, specialized, i_len=2, b_len=2, c_len=3, v_len=5, o_len=2):
    #title = unigram_get_words(data_title, 4)
    #song.append("SONG NAME: " + title.upper() + "\n")
    song = []
    #intro = "[INTRO]\n" + generate_text(model_fullset, i_len)
    #bridge = "[BRIDGE]\n" + generate_text(model_fullset, b_len)
    chorus = "[CHORUS]\n" + str(generate_text(model_fullset, c_len))
    #outro = "[OUTRO]\n" + generate_text(model_fullset, o_len)

    #intro = "[INTRO]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), i_len)
    #bridge = "[BRIDGE]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), b_len)
    #chorus = "[CHORUS]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), c_len)
    #outro = "[OUTRO]\n" + get_sequence(model_lyrics[0], model_lyrics[1], model_lyrics[2]-1, unigram_get_words(data_lyrics, 2), o_len)
    for tag in format:
        if tag == "I":
            song.append(intro)
        elif tag == "V":
            if specialized is True:
                verse = "[VERSE]\n" + get_sequence(model_verse[0], model_verse[1], model_verse[2]-1, unigram_get_words(data_lyrics, 2), v_len)
            else:
                verse = "[VERSE]\n" + generate_text(model_fullset, v_len)
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
        print(type(element))
# %%
song_general = create_song("C", specialized=False)
# %%
print_song(song_general)
# %%
text = [None, None, "Test her", "N", None]
# %%
len(text)
# %%
len(list(filter(None, text))) >= q3
# %%
