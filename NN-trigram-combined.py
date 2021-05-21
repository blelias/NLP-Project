# %%
"""
!{sys.executable} -m pip install tensorflow
!{sys.executable} -m pip install iteration_utilities
!{sys.executable} -m pip install nltk
nltk.download('punkt')
"""
from sys import float_repr_style
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from numpy import array
from iteration_utilities import deepflatten
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow import keras
from collections import defaultdict
from nltk import trigrams
import random
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import nltk
import matplotlib.pyplot as plt
from numpy import mean, std, quantile
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
avg_len = round(sum(len(x.split(" ")) for x in sents_break) / len(sents_break), 2)
length_list = [len(x.split(" ")) for x in sents_break]
print("Average line length: " + str(avg_len))
plt.boxplot(length_list)
plt.show()
# %%
def reject_outliers(data, m=2):
    return data[abs(data - mean(data)) < m * std(data)]
# %%
length_arr = array(length_list)
length_list = list(reject_outliers(length_arr))
avg_len = round(sum(length_list) / len(length_list), 2)

print("After removing outliers: " + str(avg_len))
plt.boxplot(length_list)
plt.show()
plt.hist(length_list)
plt.show()
# %%
q1 = quantile(length_arr, 0.25)
q3 = quantile(length_arr, 0.75)
avg_len = float(round(avg_len))
print("Q1 quantile: " + str(q1), "\nQ3 quantile: " + str(q3) +
"\nRounded average length: " + str(avg_len))
# %%
"""
lengths.index(max(lengths))
long_list = [x for x in sents_break if len(x.split(" ")) >= 50]
len(sents_break[4].split(" "))
length_list = [len(x.split(" ")) for x in sents_break]
length_list.index(max(length_list))
"""
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
def create_model(data, tokenizer_len=False, n_epochs=200): # Function to create the models. toklen is True if user just want the tokenizer and max_length
    tokenizer = Tokenizer() # Decleare tf.keras.preprocessing.text.Tokenizer
    tokenizer.fit_on_texts([data])
    encoded = tokenizer.texts_to_sequences([data])[0] # Transform words to integers
    v_size = len(tokenizer.word_index) + 1 # Length of vocabulary
    
    # Creating sequences of concatenated bigrams
    sequences = list()
    for x in range(2, len(encoded)):
        sequence = encoded[x-2:x+1] # Create sequences of two and two words
        sequences.append(sequence)
    max_length = max([len(seq) for seq in sequences]) # Finding the longest vector
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre') # Puts in placeholders in vectors shorter than the longest vector
    sequences = array(sequences) # Creating a 2D-array

    if tokenizer_len is True: # If the user just wants tokenizer and length, return the values
        return (tokenizer, max_length)

    # Splitting up the data into training and testing (input and output)
    X = sequences[:, :-1] 
    y = sequences[:, -1]
    y = to_categorical(y, num_classes=v_size) # Converts the vector into a binary class matrix

    # Creating the model
    model = Sequential() # A grouping of linear stacks into a keras model
    model.add(Embedding(v_size, 10, input_length=max_length-1))  # Adds a list of tensors.
    model.add(LSTM(50)) # Long short term layer with 50 units
    model.add(Dense(v_size, activation='softmax')) # Regular densely connected NN layers with softmax
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Using catagorical_crossentropy since it's technically a multinomial classificaiton problem, with Adam optimizer which is an efficient implementation of gradient descent
    model.fit(X, y, epochs=n_epochs, verbose=2)
    return model, tokenizer, max_length
# %%
# Generate sequence
def get_sequence(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    x = 0
    for i in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0] # Transform words to integeres
        encoded = pad_sequences([encoded], maxlen=max_length, padding="pre") # Pads sequence to fixed length
        yhat = model.predict_classes(encoded, verbose=0) # Predict probabilities
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat: # If the word index matches the probability
                out_word = word
                if in_text[-1] != "\n":
                    in_text += " " + out_word
                else:
                    in_text += out_word
                x += 1
                # This should be tweaked to match how beatles typically structure their songs
                if out_word.lower() in ending_words and len(in_text.split()) >= n_words/3 and x >= 5: # When the sentence is reaching its end from n_words, start looking for an ending word
                    # Potentially include line break instead for finishing lines
                    #return in_text
                    in_text += "\n"
                    x = 0
                #elif x >= n_words/2:
                elif x >= 7: # This cap can be adjusted based on longest/average line length
                        in_text += "\n"
                        x = 0
                #break
    if in_text[-1] != "\n":
        in_text += "\n"
    return in_text
# %%
def create_song(format, specialized, i_len=5, b_len=5, c_len=10, v_len=20, o_len=5):
    title = unigram_get_words(data_title, 4)
    song = []
    song.append("SONG NAME: " + title.upper() + "\n")
    if specialized is True:
        intro = "[INTRO]\n" + get_sequence(model_intro[0], model_intro[1], model_intro[2]-1, unigram_get_words(data_lyrics, 2), i_len) # Parameters: model, tokenizer, max_length
        bridge = "[BRIDGE]\n" + get_sequence(model_bridge[0], model_bridge[1], model_bridge[2]-1, unigram_get_words(data_lyrics, 2), b_len)
        chorus = "[CHORUS]\n" + get_sequence(model_chorus[0], model_chorus[1], model_chorus[2]-1, unigram_get_words(data_lyrics, 2), c_len)
        outro = "[OUTRO]\n" + get_sequence(model_outro[0], model_outro[1], model_outro[2]-1, unigram_get_words(data_lyrics, 2), o_len)
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
# Fitting the models, return model, tokenizer, and max_length (longest word vector)
#model_intro = create_model(data_intro)
#model_intro[0].save("model_intro")
model_intro = [keras.models.load_model('model_intro'), create_model(data_intro, tokenizer_len=True)[0], create_model(data_intro, tokenizer_len=True)[1]] # Loads the model and retriving the tokenizer and max_length

#model_verse = create_model(data_verse, n_epochs=500)
#model_verse[0].save("model_verse")
model_verse = [keras.models.load_model('model_verse'), create_model(data_verse, tokenizer_len=True)[0], create_model(data_verse, tokenizer_len=True)[1]]

#model_bridge = create_model(data_bridge)
#model_bridge[0].save("model_bridge")
model_bridge = [keras.models.load_model('model_bridge'), create_model(data_bridge, tokenizer_len=True)[0], create_model(data_bridge, tokenizer_len=True)[1]]

#model_chorus = create_model(data_chorus)
#model_chorus[0].save("model_chorus")
model_chorus = [keras.models.load_model('model_chorus'), create_model(data_chorus, tokenizer_len=True)[0], create_model(data_chorus, tokenizer_len=True)[1]]

#model_outro = create_model(data_outro)
#model_outro[0].save("model_outro")
model_outro = [keras.models.load_model('model_outro'), create_model(data_outro, tokenizer_len=True)[0], create_model(data_outro, tokenizer_len=True)[1]]

#model_lyrics = create_model(data_lyrics, n_epochs=500)
#model_lyrics[0].save("model_lyrics")
model_lyrics = [keras.models.load_model('model_lyrics'), create_model(data_lyrics, tokenizer_len=True)[0], create_model(data_lyrics, tokenizer_len=True)[1]]
# %%
def print_song(song):
    for element in song:
        print(element)
# %%
song_specialized = create_song("ICBOV", specialized=True, c_len=15)
song_general = create_song("ICBOV", specialized=False, c_len=15)
# %%
print_song(song_general)
# %%
# NEXT: Add grid search, look up ways to perfect the model, check typical beatles characteristics such as verse length, line length, etc, find a good random state