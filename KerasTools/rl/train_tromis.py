#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from games import tromis
from agents import dqn
from memory import uniqmemory
from callbacks import history

width, height = 6, 9
nb_frames = 2

game = tromis.Tromis(width=width, height=height, max_turn=256)

inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
x = keras.layers.Conv3D(32,3,padding='same',strides=1,activation='relu')(inp)
x = keras.layers.Conv3D(64,3,padding='same',strides=2,activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.adam(), 'logcosh')
model.summary()

m = uniqmemory.UniqMemory(memory_size=65536)
a = dqn.Agent(model=model, mem=m, num_frames = nb_frames)
history = history.HistoryLog("tromis")

a.train(game, batch_size=32, epochs=500, episodes=256,
            epsilon=[0.5, 0.0], epsilon_rate=0.1,
            gamma=0.95, reset_memory=False, observe=1024, verbose=1,
            callbacks=[])

