#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import keras
import KerasTools as KT
import numpy as np

width = height = grid_size = 12
nb_frames = 2
actions = 3

def make_crnn():
    inpc = keras.layers.Input(shape=(height, width, 1))
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inpc)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #flat  = keras.layers.Flatten()(conv2)
    pool = keras.layers.GlobalMaxPooling2D()(conv2)
    convm = keras.models.Model(inputs=inpc, outputs=pool)
    convm.summary()

    inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
    emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
    x = keras.layers.TimeDistributed(convm)(emb)
    x = keras.layers.SimpleRNN(16, return_sequences=False)(x)
    act = keras.layers.Dense(actions, activation='linear')(x)

    model = keras.models.Model(inputs=inp, outputs=act)
    model.compile(keras.optimizers.rmsprop(), 'logcosh')
    model.summary()

    return model

def make_cnn():
    inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
    gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
    #emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
    #resh = keras.layers.Reshape((nb_frames, height, width))(emb)
    perm = keras.layers.Permute((2,3,1))(gray)
    conv1 = keras.layers.Conv2D(32, (5,5), strides=(2,2), activation='relu')(perm)
    conv2 = keras.layers.Conv2D(64, (3,3), activation='relu')(conv1)
    #pool = keras.layers.GlobalMaxPooling2D()(conv2)
    flat = keras.layers.Flatten()(conv2)
    x = keras.layers.Dense(256, activation='relu')(flat)
    act = keras.layers.Dense(actions, activation='linear')(x)

    model = keras.models.Model(inputs=inp, outputs=act)
    model.compile(keras.optimizers.rmsprop(), 'logcosh')
    model.summary()

    return model

def make_dnn():
    inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
    gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
    #emb = keras.layers.Conv3D(1, 1, activation='relu', use_bias=False)(inp)
    flat = keras.layers.Flatten()(gray)
    x = keras.layers.Dense(128, activation='relu')(flat)
#    x = keras.layers.Dense(32, activation='relu')(x)
    act = keras.layers.Dense(actions, activation='linear')(x)

    model = keras.models.Model(inputs=inp, outputs=act)
    model.compile(keras.optimizers.rmsprop(), 'logcosh')
    model.summary()

    return model


model = make_cnn()

game = KT.qlearn.snake.Snake(grid_size, max_turn=64)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)
agent.train(game, batch_size=256, epochs=100, train_interval=32,
            observe=10, epsilon=0.1, #epsilon=[0.5, 0.0], epsilon_rate=0.2, 
            gamma=0.9, reset_memory=False)

