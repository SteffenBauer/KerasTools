#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')

from games import catch
from agents import dqn
from memory import uniqmemory
from callbacks import history

grid_size = 12
nb_frames = 1

game = catch.Catch(grid_size)

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
x = keras.layers.Conv3D(16,5,padding='same',strides=1,activation='relu')(inp)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.RMSprop(), 'logcosh')
model.summary()

m = uniqmemory.UniqMemory(memory_size=8192)
a = dqn.Agent(model=model, mem=m, num_frames = nb_frames)
history = history.HistoryLog("catch")

a.train(game, batch_size=32, epochs=20, episodes=256, train_interval=32,
            epsilon=[0.0, 0.0], epsilon_rate=0.25,
            gamma=0.98, reset_memory=False, observe=128, verbose=1,
            callbacks = [history])

