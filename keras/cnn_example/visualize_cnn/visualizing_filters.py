import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16

from utils import save_fig
from utils import generate_pattern


model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block4_conv1'
size = 150
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(model, layer_name, i+(j*8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[
            horizontal_start:horizontal_end,
            vertical_start:vertical_end,
            :,
        ] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results.astype(np.uint8))

fig_name = '{}_filter'.format(layer_name)
save_fig(fig_name)
