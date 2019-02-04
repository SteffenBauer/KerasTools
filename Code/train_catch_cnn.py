#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import keras
import KerasTools as KT
import numpy as np

grid_size = 16
nb_frames = 2
actions = 3

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
#gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
resh1 = keras.layers.Reshape((nb_frames, grid_size, grid_size))(emb)
perm1 = keras.layers.Permute((2,3,1))(resh1)
conv1 = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(perm1)
conv2 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
#resh2 = keras.layers.Reshape((grid_size*grid_size, 64))(conv2)
#perm2 = keras.layers.Permute((2,1))(resh2)
#rnn = keras.layers.SimpleRNN(128)(perm2)
flat = keras.layers.Flatten()(conv2)
x = keras.layers.Dense(128, activation='relu')(flat)
act = keras.layers.Dense(actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

game = KT.qlearn.catch.Catch(grid_size)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)
agent.train(game, batch_size=256, epochs=10, train_interval=32,
            epsilon=[0.5, 0.0], epsilon_rate=0.2, 
            gamma=0.99, reset_memory=False, callbacks=[KT.qlearn.callbacks.History(game.name)])

print model.layers[1].get_weights()

