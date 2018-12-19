#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import KerasTools as KT
import numpy as np

grid_size = 10
nb_frames = 5

model = keras.models.Sequential()
model.add(keras.layers.Reshape((nb_frames, grid_size, grid_size, 3), input_shape=(nb_frames, grid_size, grid_size, 3)))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Reshape((32*(grid_size-4)*(grid_size-4),))))
model.add(keras.layers.SimpleRNN(64, return_sequences=False))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3))
model.compile(keras.optimizers.rmsprop(), "logcosh")
model.summary()

game = KT.qlearn.snake.Snake(grid_size, max_turn=64)
agent = KT.qlearn.agent.Agent(model=model, memory_size=65536, nb_frames = nb_frames)

agent.train(game, batch_size=256, epochs=100, train_interval=128,
            epsilon=[0.5, 0.01], epsilon_rate=0.1, gamma=0.8, reset_memory=False)

