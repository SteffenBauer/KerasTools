#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import catch
import parallel_agent as agent
import memory

#import cProfile
#import pstats

grid_size = 12
nb_frames = 1

game = catch.Catch(grid_size)

inpc = keras.layers.Input(shape=(grid_size, grid_size, 3))
conv1 = keras.layers.Conv2D(16,3,padding='same',strides=2,activation='relu')(inpc)
conv2 = keras.layers.Conv2D(32,3,padding='same',strides=2,activation='relu')(conv1)
conv3 = keras.layers.Conv2D(64,3,padding='same',strides=2,activation='relu')(conv2)
conv4 = keras.layers.Conv2D(128,3,padding='same',strides=2,activation='relu')(conv3)
flat = keras.layers.Flatten()(conv4)
convm = keras.models.Model(inputs=inpc, outputs=flat)
convm.summary()

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
x = keras.layers.TimeDistributed(convm)(inp)
x = keras.layers.SimpleRNN(32, return_sequences=False)(x)
x = keras.layers.Dense(32, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.adam(), 'logcosh')
model.summary()

m = memory.UniqMemory(memory_size=4096)
a = agent.Agent(model=model, mem=m, num_frames = nb_frames)

#pr = cProfile.Profile()
#pr.enable()

a.train(game, batch_size=256, epochs=50, train_interval=64, turns_per_epoch=2048,
            epsilon=[1.0, 0.0], epsilon_rate=0.25,
            gamma=0.95, reset_memory=False, observe=256, verbose=1)

#pr.disable()
#stats = pstats.Stats(pr).sort_stats('cumulative')
#stats.print_stats('agent|memory|model')

