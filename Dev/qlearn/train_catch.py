#!/usr/bin/env python

import keras
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import catch
import agent
import memory

#import cProfile
#import pstats

grid_size = 16
nb_frames = 1

game = catch.Catch(grid_size)

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
perm = keras.layers.Permute((2,3,1))(gray)
conv1 = keras.layers.Conv2D(16,3,padding='same',strides=2,activation='relu')(perm)
conv2 = keras.layers.Conv2D(32,3,padding='same',strides=2,activation='relu')(conv1)
conv3 = keras.layers.Conv2D(64,3,padding='same',strides=2,activation='relu')(conv2)
conv4 = keras.layers.Conv2D(128,3,padding='same',strides=2,activation='relu')(conv3)
flat = keras.layers.Flatten()(conv4)
act = keras.layers.Dense(game.nb_actions, activation='linear')(flat)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

m = memory.UniqMemory(memory_size=65536)
a = agent.Agent(model=model, mem=m, num_frames = nb_frames)

#pr = cProfile.Profile()
#pr.enable()

a.train(game, batch_size=64, epochs=20, train_interval=8, episodes=256,
            epsilon=[0.0, 0.0], epsilon_rate=0.2,
            gamma=0.95, reset_memory=False, observe=0)

#pr.disable()
#stats = pstats.Stats(pr).sort_stats('cumulative')
#stats.print_stats('agent|memory|model')

