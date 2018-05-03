import heapq
from time import time
from random import randrange, seed
import nltk

nltk.download('punkt')

def first_k_words(text, k):
    words = set()
    i = 0
    while len(words) < k and i < len(text):
        if text[i] not in words:
            words.add(text[i])
        i = i + 1

    return list(words)

def sample_k_words(text, k):
    words = set()
    seed(time())
    while len(words) < k:
        i = randrange(0, len(text))

        if text[i] not in words:
            words.add(text[i])

    return list(words)

def top_k_word_frequencies(text, k):
    counts = {}
    heap = []

    for word in text:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    for word, freq in counts.items():
        heapq.heappush(heap, (freq, word))

    return [word for freq, word in heapq.nlargest(k, heap)]


def tokenize(text):
    return nltk.word_tokenize(text)


def tokenize_char(text):
    return list(text)


def normalize(tokens):
    return [word.lower() for word in tokens]
