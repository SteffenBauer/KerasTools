#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

img_shape = (32, 32, 3)
batch_size = 16
latent_dim = 8

input_img = keras.Input(shape=img_shape)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu', strides=(2, 2))(x)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])

x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
        
y = CustomVariationalLayer()([input_img, z_decoded])

from keras.datasets import cifar10
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)

decoder.summary()
vae.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[y_train.flatten() == 1]
x_train = x_train.astype('float32') / 255.
#x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test[y_test.flatten() == 1]
x_test = x_test.astype('float32') / 255.
#x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None))

vae.save('cifar-vae.h5')
decoder.save('cifar-decoder.h5')

