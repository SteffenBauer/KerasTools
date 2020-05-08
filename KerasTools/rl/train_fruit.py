#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')

from games import fruit
from agents import dqn
from memory import uniqmemory
from callbacks import history

game = fruit.Fruit(with_poison=True)

grid_size = game.grid_size
nb_frames = 1

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
x = keras.layers.Conv3D(32,5,padding='same',strides=1,activation='relu')(inp)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), 'logcosh')
model.summary()

params = {
    'batch_size': 32,
    'epochs': 100,
    'episodes': 256,
    'train_interval': 32,
    'epsilon': [0.1, 0.0],
    'epsilon_rate': 0.1,
    'gamma': 0.95,
    'reset_memory': False,
    'observe': 128
}

rlparams = {
    'memory': 'UniqMemory',
    'memory_size': 65536,
    'optimizer': 'RMSProp'
}

m = uniqmemory.UniqMemory(memory_size=65536)
a = dqn.Agent(model=model, mem=m, num_frames = nb_frames)
history = history.HistoryLog("fruit", {**params, **rlparams})

a.train(game, verbose=1, callbacks=[], **params)

