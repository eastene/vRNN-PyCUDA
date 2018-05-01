import heapq

import nltk

nltk.download('punkt')


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
