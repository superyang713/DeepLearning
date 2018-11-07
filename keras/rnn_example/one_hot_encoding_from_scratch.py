"""
One hot encoding from scratch
"""

import string
import numpy as np

# Word level tokenization
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
max_length = 10

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

result_shape = (
    len(samples),
    max_length,
    max(token_index.values()) + 1,
)

results = np.zeros(shape=result_shape)
for i, sample in enumerate(samples):
    for j, word in enumerate(sample.split()[:max_length]):
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results)


# Character level tokenization
characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 30

result_shape = (
    len(samples),
    max_length,
    max(token_index.values()) + 1,
)
results = np.zeros(shape=result_shape)
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

print(results)
