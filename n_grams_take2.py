#%%
import nltk
from nltk.corpus import gutenberg
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from prettytable import PrettyTable
# %%
# Unigram Language Model
model_gutenberg = defaultdict(lambda: defaultdict(lambda: 0))
def unigram(word1):
    counts = Counter(gutenberg.words())
    total_count = len(gutenberg.words())
    for words in counts:
        counts[words] /= float(total_count)
    return counts[word1]
# %%
# Bigram Language Model
## Counts
def bigram(word1, word2):
    for sentence in gutenberg.sents():
        for w1, w2 in bigrams(sentence, pad_right=True, pad_left=True):
            model_gutenberg[(w1)][w2] += 1

    ## Transforming counts into probabilities
    for w1 in model_gutenberg:
        total_count = float(sum(model_gutenberg[w1].values()))
        for w2 in model_gutenberg[w1]:
            if total_count == 0:
                model_gutenberg[w1][w2] = 0
            else:
                model_gutenberg[w1][w2] /= total_count
    return model_gutenberg[word1][word2]
# %%
# Trigram Language Model
## Counts
def trigram(word1, word2, word3):
    for sentence in gutenberg.sents():
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model_gutenberg[(w1, w2)][w3] += 1

    ## Transforming counts into probabilities
    for w1_w2 in model_gutenberg:
        total_count = float(sum(model_gutenberg[w1_w2].values()))
        for w3 in model_gutenberg[w1_w2]:
            if total_count == 0:
                model_gutenberg[w1_w2][w3] = 0
            else:
                model_gutenberg[w1_w2][w3] /= total_count
    return model_gutenberg[word1, word2][word3]
# %%
print("The probability of word occuring: ", unigram("am"))
print("The probabiltiy of word2 given word1: ", bigram("I", "am"))
print("The probability of word3 given word1 and word2: ", trigram(None, None, "The"))