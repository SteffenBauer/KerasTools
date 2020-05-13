#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')

from games import tromis
from agents import ddqn
from memory import uniqmemory
from callbacks import history

width, height = 6, 9
nb_frames = 1

game = tromis.Tromis(width, height, max_turn=64)

inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
x = keras.layers.Conv3D(64,3,padding='same',strides=2,activation='relu')(inp)
x = keras.layers.Conv3D(128,3,padding='same',strides=2,activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.Adam(), 'logcosh')
model.summary()

params = {
    'batch_size': 32,
    'epochs': 500,
    'episodes': 32,
    'target_sync': 256,
    'epsilon_start': 0.5,
    'epsilon_decay': 0.9,
    'epsilon_final': 0.0,
    'gamma': 0.9,
    'reset_memory': False,
    'observe': 1024
}

rlparams = {
    'rl.memory': 'UniqMemory',
    'rl.memory_size': 65536,
    'rl.optimizer': 'Adam',
    'rl.with_target': True,
    'rl.nb_frames': nb_frames
}

gameparams = {
    'game.width': game.width,
    'game.height': game.height,
    'game.max_turn': game.max_turn
}

memory = uniqmemory.UniqMemory(memory_size=rlparams['rl.memory_size'])
agent = ddqn.Agent(model, memory, with_target=rlparams['rl.with_target'])
history = history.HistoryLog("tromis", {**params, **rlparams, **gameparams})

agent.train(game, verbose=1, callbacks=[history], **params)

