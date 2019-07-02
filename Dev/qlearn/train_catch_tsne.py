#!/usr/bin/env python3

import pickle

import keras
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import catch
import agent
import memory
import callbacks

grid_size = 12
nb_frames = 1

game = catch.Catch(grid_size)


inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
gray = keras.layers.TimeDistributed(keras.layers.Lambda(lambda t:t[...,0]*0.1 + t[...,1]*0.3 + t[...,2]*0.6))(inp)
perm = keras.layers.Permute((2,3,1))(gray)
conv1 = keras.layers.Conv2D(8,3,padding='same',activation='relu')(perm)
conv2 = keras.layers.Conv2D(16,3,padding='same',activation='relu')(conv1)
flat = keras.layers.Flatten()(conv2)
x = keras.layers.Dense(32, activation='relu')(flat)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

m = memory.UniqMemory(memory_size=65536)
a = agent.Agent(model=model, mem=m, num_frames = nb_frames)

class SaveModel(callbacks.Callback):
    def __init__(self):
        pass
    def epoch_end(self, *args):
        model.save("./train_data/catch_{:02d}.h5".format(args[2]))

model.save("./train_data/catch_00.h5")
a.train(game, batch_size=32, epochs=50, train_interval=32, episodes=256,
            epsilon=0.0, gamma=0.95, reset_memory=False, observe=0, verbose=2,
            callbacks=[SaveModel()])

with open("./train_data/catch_mem.pkl", 'wb') as fp:
          pickle.dump(m.memory, fp)

