#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections

import keras
import numpy as np

max_features = 10000 # Top most frequent words to consider
maxlen = 200         # Cut texts after this number of words

print('Load data...')
(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(num_words=max_features)

print('Pad sequences (samples x time)')
x_train = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
y_train = keras.utils.to_categorical(train_labels)
x_test = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=maxlen)
y_test = keras.utils.to_categorical(test_labels)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

mapping = collections.Counter(train_labels)
weights = {k: float(len(train_labels)) / float((len(mapping)) * mapping[k]) for k in mapping}

def build_dnn():
    inp = keras.layers.Input(shape=(maxlen, ))
    emb = keras.layers.Embedding(max_features, 4, mask_zero=False)(inp)
    flt = keras.layers.Flatten()(emb)
    dns = keras.layers.Dense(64, activation='relu')(flt)
    out = keras.layers.Dense(46, activation='softmax')(dns)

    model = keras.models.Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

def build_rnn():
    inp = keras.layers.Input(shape=(maxlen, ))
    emb = keras.layers.Embedding(max_features, 4, mask_zero=True)(inp)
    rnn = keras.layers.GRU(64)(emb)
    out = keras.layers.Dense(46, activation='softmax')(rnn)
    
    model = keras.models.Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

def build_attn():

    inp = keras.layers.Input(shape=(4,))
    att = keras.layers.Dense(4, activation='softmax')(inp)
    atp = keras.layers.multiply([inp, att])
    atp = keras.layers.Dense(64)(atp)
    atp = keras.layers.Activation('relu')(atp)
    attention = keras.models.Model(inputs=inp, outputs=atp)

    inp = keras.layers.Input(shape=(maxlen, ))
    emb = keras.layers.Embedding(max_features, 4, mask_zero=False)(inp)
    att = keras.layers.TimeDistributed(attention)(emb)
    flt = keras.layers.Flatten()(att)
    out = keras.layers.Dense(46, activation='softmax')(flt)
    model = keras.models.Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

print('Build model...')
model = build_attn()
model.summary()

print('Train...')
history = model.fit(x_train, y_train, class_weight = None,
          batch_size=256, epochs=25, validation_split=0.1)

print('Build and train final model...')
model = build_attn()
final_epochs = int(np.argmin(history.history['val_loss'])+1)
model.fit(x_train, y_train, batch_size=256, epochs=final_epochs, class_weight = None)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss", test_loss)
print("Test accuracy", test_acc)

