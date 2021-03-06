import os
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from utils import plot_loss_and_accuracy


# Some useful parameters
n_timesteps = 100
n_samples_train = 20000
n_samples_val = 5000
max_words = 10000   # Consider only the top 10000 words in the dataset.

# Parse data
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
train_folder = os.path.join(data_folder, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_folder, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname)) as fin:
                texts.append(fin.read())

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# Tokenizing the texts of raw IMDB data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=n_timesteps)

labels = np.asarray(labels)
print('The shape of data tensor is:', data.shape)
print('The shape of label tensor is:', labels.shape)

# Split train and val
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

X_train = data[:n_samples_train]
y_train = labels[:n_samples_train]
X_val = data[n_samples_train:n_samples_train + n_samples_val]
y_val = labels[n_samples_train:n_samples_train + n_samples_val]

print('The number of training samples are:', X_train.shape[0])
print('The number of validation samples are:', X_val.shape[0])

# Parse GloVe word-embeddings file
glove_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)),
    'data',
    'glove_pre_trained_embedding'
)
embeddings_index = {}
with open(os.path.join(glove_folder, 'glove.6B.100d.txt')) as fin:
    for line in fin:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found {} word vectors.'.format(len(embeddings_index)))

# Preparing the GloVe word-embedding matrix
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, index in word_index.items():
    if index < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_matrix is not None:
            embedding_matrix[index] = embedding_vector

# Build the model
model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=n_timesteps))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Load the pre-trained GloVe embedding in the model
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Compile
model.compile(
    optimizer=optimizers.RMSprop(1e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
)

# Visualization
plot_loss_and_accuracy(history)
