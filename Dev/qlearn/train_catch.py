#!/usr/bin/env python

import keras
import tensorflow as tf

import catch
import agent
import memory

tf.logging.set_verbosity(tf.logging.ERROR)

#import cProfile
#import pstats

grid_size = 16
nb_frames = 2

game = catch.Catch(grid_size)

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
gray = keras.layers.Lambda(lambda t:t[...,0]*0.3 + t[...,1]*0.6 + t[...,2]*0.1)(inp)
flat = keras.layers.Flatten()(gray)
x = keras.layers.Dense(128, activation='relu')(flat)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

m = memory.UniqMemory(memory_size=65536)
a = agent.Agent(model=model, mem=m, num_frames = nb_frames)

#pr = cProfile.Profile()
#pr.enable()

a.train(game, batch_size=256, epochs=20, train_interval=32, episodes=256,
            epsilon=[0.5, 0.0], epsilon_rate=0.2,
            gamma=0.95, reset_memory=False)

#pr.disable()
#stats = pstats.Stats(pr).sort_stats('cumulative')
#stats.print_stats('agent|memory|model')

