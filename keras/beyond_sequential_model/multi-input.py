import numpy as np

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input


text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# Get the dummy data
n_samples = 1000
n_words = 100
text = np.random.randint(
    1, text_vocabulary_size, size=(n_samples, n_words)
)
question = np.random.randint(
    1, question_vocabulary_size, size=(n_samples, n_words)
)
answers = np.random.randint(0, 1, size=(n_samples, answer_vocabulary_size))

# First Input
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

# Second Input
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(
    question_input
)
encoded_question = layers.LSTM(16)(embedded_question)

# Concatenate output
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(
    concatenated
)

# Compile
model = models.Model([text_input, question_input], answer)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc'],
)

# Fit(train) model-------------First method
# model.fit([text, question], answers, epochs=10, batch_size=128)

# Fit(train) model-------------Second method
model.fit(
    {'text': text, 'question': question},
    answers,
    epochs=10,
    batch_size=128
)
