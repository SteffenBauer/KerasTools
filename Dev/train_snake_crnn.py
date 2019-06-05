#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import KerasTools as KT
import numpy as np

grid_size = 12
nb_frames = 4

inpc = keras.layers.Input(shape=(grid_size, grid_size, 3))
conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inpc)
#mpool = keras.layers.MaxPooling2D(3)(conv1)
conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
gpool = keras.layers.GlobalMaxPooling2D()(conv2)
#resh = keras.layers.Reshape((32*(grid_size-4)*(grid_size-4),))(conv2)
convm = keras.models.Model(inputs=inpc, outputs=gpool)
convm.summary()

inp = keras.layers.Input(shape=(None, grid_size, grid_size, 3))
x = keras.layers.TimeDistributed(convm)(inp)
x = keras.layers.SimpleRNN(32, return_sequences=False)(x)
#x = keras.layers.GlobalMaxPooling1D()(x)
#x = keras.layers.Dense(64, activation='relu')(x)
act = keras.layers.Dense(3, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

game = KT.qlearn.snake.Snake(grid_size, max_turn=64)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)
agent.train(game, batch_size=256, epochs=10, train_interval=128,
            epsilon=[0.5, 0.0], epsilon_rate=0.5, 
            gamma=0.8, reset_memory=False)

