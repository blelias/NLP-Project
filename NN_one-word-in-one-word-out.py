# %%
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
df = pd.read_csv("cleaned_dataset.csv")
# %%
data = df.verse_cleaned[0]
data = joinedlist = ".".join(list(df.verse_cleaned.dropna()))
# %%
# Model 1 (One-word-in, one-word-out)
tokenizer = tf.keras.preprocessing.text.Tokenizer() # Encodes tokens into integers
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0] 
# %%
vocab_size = len(tokenizer.word_index) + 1
# %%
# Creaing sequences
sequences = list()
for i in range(1, len(encoded)): # For every word
    sequence = encoded[i-1:i+1] # The sequence is two consequent words
    sequences.append(sequence)
# %%
# Creating inputs and outputs
sequences = array(sequences) # Put into 2d-array
X, y = sequences[:,0], sequences[:, 1] # X = first col, y = second col
# %%
y = to_categorical(y, num_classes=vocab_size) # One-hot encoding
# %%
# Model uses softmax for output
model = Sequential() # A grouping of linear stacks into a keras model
model.add(Embedding(vocab_size, 10, input_length=1)) # Adds a list of tensors.
# Embedding is the first layer, input dim = vocab size, output dim = 10
model.add(LSTM(50)) # Long short term layer with 50 units
model.add(Dense(vocab_size, activation="softmax"))
# Regular densely connected NN layers with softmax
# %%
print(model.summary())
# %%
# Using catagorical_crossentropy since it's technically a multinomial classificaiton problem
# Adam optimizer is an efficient implementation of gradient descent
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=500, verbose=2) # Verbose visualizes the training

# %%
# Evaluation
in_text = "and"
encoded = tokenizer.texts_to_sequences([in_text])[0]
encoded = array(encoded)
yhat = model.predict_classes(encoded, verbose=0)
for word, index in tokenizer.word_index.items():
    if index == yhat:
        print(word)
# %%
def seq_generator(model, tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    for i in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = array(encoded)
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
            else:
                pass
        in_text, result = out_word, result + " " + out_word
    return result
# %%
verse = seq_generator(model, tokenizer, 'play', 15)
# %%
verse
# %%
