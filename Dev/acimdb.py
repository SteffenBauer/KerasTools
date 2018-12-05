#!/usr/bin/env python
# coding: utf-8

import keras
import string
import random
import numpy as np

translate = { c:i+1 for i,c in enumerate(string.ascii_lowercase)}
translate[' '] = 1+len(string.ascii_lowercase)
translate['PAD'] = 0
numtokens = len(translate)

train_data = []
train_labels = []
with open('train.dat', 'r') as fp:
    for line in fp:
        label, text = line.split(',')
        train_labels.append(int(label))
        tokenized_text = [translate[c] for c in text.strip()]
        train_data.append(tokenized_text)

maxlen = max(len(t) for t in train_data)
maxlen = 2000

train_set = zip(train_labels, train_data)
random.shuffle(train_set)

train_labels = np.asarray([t[0] for t in train_set])
train_data = [t[1] for t in train_set]
train_input = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(numtokens, 16, input_shape=(maxlen, ), mask_zero=False))
model.add(keras.layers.SeparableConv1D(32, kernel_size=7, activation='relu', use_bias=False))
model.add(keras.layers.MaxPooling1D(7))
model.add(keras.layers.SeparableConv1D(64, kernel_size=7, activation='relu', use_bias=False))
#model.add(keras.layers.Masking())
#model.add(keras.layers.GRU(32, return_sequences=False))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

model.summary()

print('Train...')
history = model.fit(train_input, train_labels,
          batch_size=256, epochs=100, validation_split=0.2)

