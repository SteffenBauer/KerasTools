#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate
 
vocab_size = 10000
 
pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size, start_char=start_id,
                                                                        oov_char=oov_id, index_from=index_offset)
 
word2idx = tf.keras.datasets.imdb.get_word_index()
 
idx2word = {v + index_offset: k for k, v in word2idx.items()}
 
idx2word[pad_id] = '<PAD>'
idx2word[start_id] = '<START>'
idx2word[oov_id] = '<OOV>'
 
max_len = 200
rnn_cell_size = 128
 
x_train = sequence.pad_sequences(x_train,
                                 maxlen=max_len,
                                 truncating='post',
                                 padding='post',
                                 value=pad_id)
x_test = sequence.pad_sequences(x_test, maxlen=max_len,
                                truncating='post',
                                padding='post',
                                value=pad_id)
                                
class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
 
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
 
        return context_vector, attention_weights

sequence_input = Input(shape=(max_len,), dtype='int32')
 
embedded_sequences = keras.layers.Embedding(vocab_size, 128, input_length=max_len)(sequence_input)

import os
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                     (rnn_cell_size,
                                      dropout=0.3,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_activation='relu',
                                      recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)
 
lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional \
    (tf.keras.layers.LSTM
     (rnn_cell_size,
      dropout=0.2,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(lstm)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

attention = Attention(16)

context_vector, attention_weights = attention(lstm, state_h)
 
output = keras.layers.Dense(1, activation='sigmoid')(context_vector)
 
model = keras.Model(inputs=sequence_input, outputs=output)
 
# summarize layers
print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
 
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=0,
                                                        patience=1,
                                                        verbose=0, mode='auto')

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=200,
                    validation_split=.3, verbose=1, callbacks=[early_stopping_callback])

result = model.evaluate(x_test, y_test)
print(result)

