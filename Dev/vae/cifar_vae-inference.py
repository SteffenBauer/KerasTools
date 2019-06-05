#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

decoder = keras.models.load_model('cifar-decoder.h5')
decoder.summary()

n = 8
digit_size = 32
batch_size = 16
latent_dim = 8

figure = np.zeros((digit_size * n, digit_size * n, 3))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi, 1, 1,1,1,1,1]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size, 3)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

