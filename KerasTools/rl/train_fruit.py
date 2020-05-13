#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')

from games import fruit
from agents import ddqn
from memory import uniqmemory
from callbacks import history

grid_size = 12
nb_frames = 1

game = fruit.Fruit(grid_size, with_poison=True)

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
x = keras.layers.Conv3D(32,3,padding='same',strides=2,activation='relu')(inp)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.RMSprop(), 'logcosh')
model.summary()

params = {
    'batch_size': 32,
    'epochs': 100,
    'episodes': 32,
    'target_sync': 96,
    'epsilon_start': 0.5,
    'epsilon_decay': 0.75,
    'epsilon_final': 0.0,
    'gamma': 0.95,
    'reset_memory': False,
    'observe': 128
}

rlparams = {
    'rl.memory': 'UniqMemory',
    'rl.memory_size': 65536,
    'rl.optimizer': 'RMSprop',
    'rl.with_target': True,
    'rl.nb_frames': nb_frames
}

gameparams = {
    'game.grid_size': game.grid_size,
    'game.with_poison': game.with_poison,
    'game.penalty': game.penalty,
    'game.max_turn': game.max_turn
}

memory = uniqmemory.UniqMemory(memory_size=rlparams['rl.memory_size'])
agent = ddqn.Agent(model, memory, with_target=rlparams['rl.with_target'])
history = history.HistoryLog("fruit", {**params, **rlparams, **gameparams})

agent.train(game, verbose=1, callbacks=[history], **params)

