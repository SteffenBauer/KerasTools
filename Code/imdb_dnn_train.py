#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import keras
import imdb_cache

if not os.path.exists('weights'):
    os.makedirs('weights/')

max_features = 50000
maxlen = 100
batch_size = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb_cache.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post',truncating='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post',truncating='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = keras.models.Sequential()
model.add(keras.layers.Embedding(max_features, 2, input_shape=(maxlen, )))
model.add(keras.layers.Conv1D(2, 7, activation='relu', padding='valid'))
#model.add(keras.layers.MaxPooling1D(pool_size=5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

model.summary()

print('Train...')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=30,
          validation_split=0.2,
          callbacks=[keras.callbacks.ModelCheckpoint('weights/imdb_dnn.h5', monitor='val_loss',
                                     save_best_only=True, save_weights_only=False)])

with open('imdb_dnn.hist', 'w') as fp:
    json.dump(history.history, fp)

model = keras.models.load_model('weights/imdb_dnn.h5')

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

