#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import imdb_cache
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

max_features = 50000
maxlen = 100

entry = 21

(x_train, y_train), (x_test, y_test) = imdb_cache.load_data(num_words=max_features)

raw_word_index = imdb_cache.get_word_index()
word_index = {v+3:k for k,v in raw_word_index.items()}
word_index[0] = '-PAD-'
word_index[1] = '-START-'
word_index[2] = '-UNK-'

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post',truncating='post')
x_test =  keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post',truncating='post')
model = keras.models.load_model('weights/imdb_dnn.h5')
model.summary()

conv_out = model.layers[1].output
conv_model = keras.models.Model(inputs = model.input, outputs = conv_out)

result = model.predict(x_test[entry:entry+1])
conv_result = conv_model.predict(x_test[entry:entry+1])

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print()

for i in range(len(conv_result[0])):
    snippet = " ".join(word_index.get(w, 2) for w in x_test[entry][i:i+7])
    r1,r2 = conv_result[0][i]
    print("{:4.4f} {:4.4f} {}".format(r1,r2, snippet))
print()
print('Reconstructed review text:')
print('--------------------------------')
print(" ".join(word_index.get(w, 2) for w in x_test[entry]))
print('--------------------------------')
print()
print(result[0][0], y_test[entry])

