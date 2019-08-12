#!/usr/bin/env python3

import keras
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import catch
import agent

grid_size = 12
nb_frames = 1

game = catch.Catch(grid_size)

import keras.backend as K

def policy_gradient_loss(Returns):
    def modified_crossentropy(action,action_probs):
        cost = K.categorical_crossentropy(action,action_probs,from_logits=False,axis=1 * Returns)
        return K.mean(cost)
    return modified_crossentropy

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
flt = keras.layers.Flatten()(inp)
den = keras.layers.Dense(64, activation='relu')(flt)
act = keras.layers.Dense(game.nb_actions, activation='softmax')(den)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.rmsprop(), loss=modified_crossentropy)
model.summary()

a = agent.Agent(model=model, num_frames=nb_frames)
a.train(game, episodes=10, gamma=0.95)

