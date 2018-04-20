import heapq


def top_k_word_frequencies(text, k):
    counts = {}
    heap = []

    for word in text.split(' ').strip('.;,:\n'):
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    for word, freq in counts:
        heapq.heappush(heap, (freq, word))

    return [word for freq, word in heapq.nlargest(k, heap)]
