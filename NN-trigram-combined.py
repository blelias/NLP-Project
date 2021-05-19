# %%
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
# %%
df = pd.read_csv("cleaned_dataset_2.csv")
data = ".".join(list(df.verse_cleaned_2[0:50].dropna()))
# %%
tokenizer = Tokenizer() # Decleare tf.keras.preprocessing.text.Tokenizer
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0] # Transform words to integers
v_size = len(tokenizer.word_index) + 1 # Length of vocabulary
# %%
# Creating sequences of concatenated bigrams
sequences = list()
for x in range(2, len(encoded)):
    sequence = encoded[x-2:x+1] # Create sequences of two and two words
    sequences.append(sequence)
max_length = max([len(seq) for seq in sequences]) # Finding the longest vector
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre') # Puts in placeholders in vectors shorter than the longest vector
# %%
sequences = array(sequences) # Creating a 2D-array
X = sequences[:, :-1] # Splitting up the data into training and testing (input and output)
y = sequences[:, -1]
y = to_categorical(y, num_classes=v_size) # Converts the vector into a binary class matrix
# %%
# Creating the model
model = Sequential() # A grouping of linear stacks into a keras model
model.add(Embedding(v_size, 10, input_length=max_length-1))  # Adds a list of tensors.
model.add(LSTM(50)) # Long short term layer with 50 units
model.add(Dense(v_size, activation='softmax')) # Regular densely connected NN layers with softmax
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Using catagorical_crossentropy since it's technically a multinomial classificaiton problem, with Adam optimizer which is an efficient implementation of gradient descent
# %%
# Fitting model
model.fit(X, y, epochs=100, verbose=2)
# %%
# Generate sequence
def get_sequence(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0] # Transform words to integeres
        encoded = pad_sequences([encoded], maxlen=max_length, padding="pre") # Pads sequence to fixed length
        yhat = model.predict_classes(encoded, verbose=0) # Predict probabilities
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat: # If the word index matches the probability
                out_word = word
                break
        in_text += " " + out_word
    return in_text
# %%
def create_song(format):
    song = []
    intro = "[INTRO] " + get_sequence(model, tokenizer, max_length-1, 'are you', 5)
    chorus = "[CHORUS] " + get_sequence(model, tokenizer, max_length-1, 'are you', 5)
    outro = "[OUTRO] " + get_sequence(model, tokenizer, max_length-1, 'are you', 5)

    for tag in format:
        if tag == "I":
            song.append(intro)
        elif tag == "V":
            verse = "[VERSE] " + get_sequence(model, tokenizer, max_length-1, 'are you', 5)
            song.append(verse)
        elif tag == "C":
            song.append(chorus)
        elif tag == "O":
            song.append(outro)
    return song

# %%
for element in create_song("IVCVO"):
    print(element)
# %%
