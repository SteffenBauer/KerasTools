#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import keras
import KerasTools as KT
import numpy as np

import cProfile
import pstats

grid_size = 16
nb_frames = 2
actions = 3

def make_crnn():
    inpc = keras.layers.Input(shape=(grid_size, grid_size, 3))
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inpc)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    flat  = keras.layers.Flatten()(conv2)
    convm = keras.models.Model(inputs=inpc, outputs=flat)
    convm.summary()

    inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
    x = keras.layers.TimeDistributed(convm)(inp)
    x = keras.layers.SimpleRNN(128, return_sequences=False)(x)
    act = keras.layers.Dense(actions, activation='linear')(x)

    model = keras.models.Model(inputs=inp, outputs=act)
    model.compile(keras.optimizers.rmsprop(), 'logcosh')
    model.summary()

    return model

def make_cnn():
    inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
    #gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
    emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
    resh = keras.layers.Reshape((nb_frames, grid_size, grid_size))(emb)
    perm = keras.layers.Permute((2,3,1))(resh)
    conv1 = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(perm)
    conv2 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    flat = keras.layers.Flatten()(conv2)
    x = keras.layers.Dense(128, activation='relu')(flat)
    act = keras.layers.Dense(actions, activation='linear')(x)

    model = keras.models.Model(inputs=inp, outputs=act)
    model.compile(keras.optimizers.rmsprop(), 'logcosh')
    model.summary()

    return model

def make_dnn():
    inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
    gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
    #emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
    flat = keras.layers.Flatten()(gray)
    x = keras.layers.Dense(128, activation='relu')(flat)
    #x = keras.layers.Dense(128, activation='relu')(x)
    act = keras.layers.Dense(actions, activation='linear')(x)

    model = keras.models.Model(inputs=inp, outputs=act)
    model.compile(keras.optimizers.rmsprop(), 'logcosh')
    model.summary()

    return model


model = make_dnn()

game = KT.qlearn.catch.Catch(grid_size)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)

pr = cProfile.Profile()
pr.enable()

agent.train(game, batch_size=256, epochs=10, train_interval=32, episodes=256,
            epsilon=0.1, #epsilon=[0.5, 0.0], epsilon_rate=0.2, 
            gamma=0.95, reset_memory=False)

pr.disable()
stats = pstats.Stats(pr).sort_stats('cumulative')
stats.print_stats('agent|memory|model')

