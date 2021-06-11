# %%
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from collections import Counter, defaultdict
from nltk import trigrams
import random
# %%
df = pd.read_csv("cleaned_dataset.csv")
# %%
data = df.lyrics_cleaned[0]
joinedlist = ".".join(list(df.chorus_cleaned.dropna()))
# %%
sample_text = joinedlist

sents = sent_tokenize(sample_text)
print('number of sentences: ', len(sents))

corpus_sents = []

for sent in sents:
    corpus_sents.append(word_tokenize(sent))
    
model_sample = defaultdict(lambda: defaultdict(lambda: 0))
# %%
corpus_sents
# %%
#iterate over the trigrams in corpus sentences and assign a score 1 to the existing ones
for sentence in corpus_sents:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model_sample[(w1, w2)][w3] += 1
# %%
model_sample
# %%
for w1_w2 in model_sample:
    total_count = float(sum(model_sample[w1_w2].values()))
    if total_count == 0:
        print(w1_w2, ':', total_count, ':', model_sample[w1_w2])

    for w3 in model_sample[w1_w2]:
        if total_count == 0:
            model_sample[w1_w2][w3] = 0
        else:
            model_sample[w1_w2][w3] /= total_count

#print(model_sample)
# %%
# Generating text
# Generating some random text using trigrams
text = [None, None]

sentence_finished = False

while not sentence_finished:
    r = random.random()
    accumulator = .0
    
    for word in model_sample[tuple(text[-2:])].keys():
        accumulator += model_sample[tuple(text[-2:])][word]
        if accumulator >= r:
            text.append(word)
            break
        else:
            pass
    if text[-2:] == [None, None]:
        sentence_finished = True
 
# %%
sentence = ' '.join([t for t in text if t])
# %%
sentence
# %%
