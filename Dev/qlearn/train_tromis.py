#!/usr/bin/env python

import json

import keras
import tensorflow as tf

import tromis
import agent
import memory

#import cProfile
#import pstats

tf.logging.set_verbosity(tf.logging.ERROR)

width, height, nb_frames = 5, 8, 4

game = tromis.Tromis(width=width, height=height, max_turn=64)

'''
inpc = keras.layers.Input(shape=(height, width, 3))
conv1 = keras.layers.Conv2D(8, 3, activation='relu', strides=2, padding='same')(inpc)
#pool1 = keras.layers.AveragePooling2D(2)(conv1)
conv2 = keras.layers.Conv2D(16, 3, activation='relu', strides=2, padding='same')(conv1)
#pool2 = keras.layers.AveragePooling2D(2)(conv2)
conv3 = keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(conv2)
#gpool = keras.layers.GlobalAveragePooling2D()(conv3)
flt = keras.layers.Flatten()(conv3)
convm = keras.models.Model(inputs=inpc, outputs=flt)
convm.summary()

inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
#gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
#exp = keras.layers.Reshape((nb_frames, height, width, 1))(gray)
x = keras.layers.TimeDistributed(convm)(inp)
x = keras.layers.SimpleRNN(64, return_sequences=False)(x)
#x = keras.layers.Reshape((4*128,))(x)
act = keras.layers.Dense(5, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()
'''


inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
#gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
#perm = keras.layers.Permute((2,3,1))(gray)
conv1 = keras.layers.Conv3D(64, 3, activation='relu', strides=(1,2,2), padding='same')(inp)
#conv2 = keras.layers.Conv2D(16, 3, strides=2, activation='relu')(conv1)
flat = keras.layers.Flatten()(conv1)
#avg = keras.layers.GlobalAveragePooling2D()(conv)
#x = keras.layers.Dense(64, activation='relu')(flat)
act = keras.layers.Dense(game.nb_actions, activation='linear')(flat)
model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()


m = memory.UniqMemory(memory_size=65536)
a = agent.Agent(model=model, mem=m, num_frames = nb_frames)

#pr = cProfile.Profile()
#pr.enable()

history = a.train(game, batch_size=256, epochs=100, train_interval=32, episodes=256,
            epsilon=0.0, # [0.5, 0.0], epsilon_rate=0.1,
            gamma=0.95, reset_memory=False)

with open('tromis_g0975_e100.hist','w') as fp:
    json.dump(history, fp)

#pr.disable()
#stats = pstats.Stats(pr).sort_stats('cumulative')
#stats.print_stats('agent|memory|model')

