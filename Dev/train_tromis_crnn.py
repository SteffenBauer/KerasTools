#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import KerasTools as KT
import numpy as np

width = 5
height = 8
nb_frames = 12

inpc = keras.layers.Input(shape=(height, width, 3))
conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inpc)
conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
gpool = keras.layers.GlobalMaxPooling2D()(conv2)
convm = keras.models.Model(inputs=inpc, outputs=gpool)
convm.summary()

inp = keras.layers.Input(shape=(None, height, width, 3))
x = keras.layers.TimeDistributed(convm)(inp)
x = keras.layers.SimpleRNN(32, return_sequences=False)(x)
act = keras.layers.Dense(5, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

game = KT.qlearn.tromis.Tromis(width=width,height=height)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)
agent.train(game, batch_size=256, epochs=20, train_interval=128,
            epsilon=[0.5, 0.0], epsilon_rate=0.5,
            gamma=0.5, reset_memory=False)

