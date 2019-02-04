#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import keras
import KerasTools as KT
import numpy as np

width = 5
height = 8
nb_frames = 2
actions = 5

inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
#gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
resh = keras.layers.Reshape((nb_frames, height, width))(emb)
perm = keras.layers.Permute((2,3,1))(resh)
conv1 = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(perm)
conv2 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
flat = keras.layers.Flatten()(conv2)
x = keras.layers.Dense(128, activation='relu')(flat)
act = keras.layers.Dense(actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

game = KT.qlearn.tromis.Tromis(width=width, height=height, max_turn=128)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)
agent.train(game, batch_size=256, epochs=200, train_interval=32,
            epsilon=[0.5, 0.0], epsilon_rate=0.05, 
            gamma=0.95, reset_memory=False, callbacks = [KT.qlearn.callbacks.History(game.name)])

print model.layers[1].get_weights()

