#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SGD       19s/epoch
# SGD-mom   19s/epoch
# AdaBound  23s/epoch
# Adam      23s/epoch
# RMSprop   21s/epoch

import keras
from keras.applications import densenet
import json
from adabound import AdaBound

batch_size = 64
nb_epoch = 30

img_rows, img_cols = 32, 32
img_channels = 3

model = densenet.DenseNet([1,2,3,2],
    include_top=True, weights=None, pooling='avg',
    input_shape=(img_rows, img_cols, img_channels), classes=10)
model.compile(loss='categorical_crossentropy',
              optimizer=AdaBound(), #keras.optimizers.SGD(momentum=0.9),
              metrics=['acc'])
model.summary()

(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = densenet.preprocess_input(trainX)
testX = densenet.preprocess_input(testX)

Y_train = keras.utils.to_categorical(trainY, 10)
Y_test = keras.utils.to_categorical(testY, 10)

history = model.fit(trainX, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1)

with open('./cifar10_densenet_adabound.hist', 'w') as fp:
    json.dump(history.history, fp)

