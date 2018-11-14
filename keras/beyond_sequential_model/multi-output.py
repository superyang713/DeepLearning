from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input


vocabulary_size = 50000
n_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalAvgPool1D()(x)

age_predication = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(
    n_income_groups,
    activation='softmax',
    name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = models.Model(
    posts_input,
    [age_predication, income_prediction, gender_prediction],
)

# Compilation options of a multi-output model: loss weighting
# Can also be done in a dictionary way.
model.compile(
    optimizer='rmsprop',
    loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
    loss_weights=[0.25, 1., 10.]
)
