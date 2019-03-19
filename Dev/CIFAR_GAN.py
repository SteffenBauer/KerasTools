#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import json

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

tf.logging.set_verbosity(tf.logging.ERROR)

def build_generator(latent_dim, channels):

    noise = keras.layers.Input(shape=(latent_dim,))
    
    x = keras.layers.Dense(latent_dim * 4 * 4)(noise)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Reshape((4, 4, latent_dim))(x)
        
    x = keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)

    x = keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    
    x = keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)

    x = keras.layers.Conv2D(channels, kernel_size=5, padding="same")(x)
    img = keras.layers.Activation("tanh")(x)

    return noise, img

def build_discriminator(img_shape):

    img = keras.layers.Input(shape=img_shape)
    x = keras.layers.GaussianNoise(0.05)(img)

    x = keras.layers.Conv2D(32, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.AveragePooling2D(2)(x)
    
    x = keras.layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.AveragePooling2D(2)(x)

    x = keras.layers.Conv2D(256, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU(1)(x)
    x = keras.layers.AveragePooling2D(2)(x)

    x = keras.layers.Conv2D(512, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.AveragePooling2D(2)(x)

    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Flatten()(x)
    validity = keras.layers.Dense(1, activation='sigmoid')(x)

    return img, validity
    
def rounded_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 32

        optimizer = keras.optimizers.Adam(0.0002) #, 0.5) #, clipvalue=1.0, decay=1e-8)
        #optimizer = keras.optimizers.SGD(0.002, momentum=0.8, nesterov=True, clipvalue=1.0, decay=1e-7)
        
        # Build and compile the discriminator
        inp_discriminator, out_discriminator = build_discriminator(self.img_shape)
        self.discriminator = keras.models.Model(inp_discriminator, out_discriminator, name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[rounded_binary_accuracy])

        # Build the frozen discriminator copy
        self.frozen_discriminator = keras.engine.network.Network(inp_discriminator, out_discriminator, name='frozen_discriminator')
        self.frozen_discriminator.trainable = False

        # Build the generator
        inp_generator, out_generator = build_generator(self.latent_dim, self.channels)
        self.generator = keras.models.Model(inp_generator, out_generator, name='generator')

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        #optimizer = keras.optimizers.Adam(0.0002, 0.5) #, clipvalue=1.0, decay=1e-8)
        
        self.combined = keras.models.Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.frozen_discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.generator.summary()
        self.discriminator.summary()
        self.combined.summary()

    def init(self, batch_size=128, cls=1):
        self.cls = cls
        self.batch_size = batch_size

        # Load the dataset
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data() #(label_mode='fine')
        
        X_train = np.append(X_train, X_test, axis=0)
        self.Y_train = np.append(Y_train, Y_test, axis=0)
        X_train = X_train[self.Y_train.flatten() == self.cls]

        # Rescale -1 to 1
        self.X_train = X_train / 127.5 - 1.0

        self.loss_discriminator = []
        self.acc_discriminator = []
        self.loss_generator = []
        self.epoch = 0
        
    def train(self, epochs, save_interval=50):
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(self.batch_size, 1))
        valid_fake = np.ones((self.batch_size, 1))
        
        # History arrays
        d_losses = []
        d_accs = []
        g_losses = []
        start_epoch = self.epoch
        
        for self.epoch in range(self.epoch, start_epoch+epochs+1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            imgs = self.X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid_fake)

            # Plot the progress
            d_losses.append(d_loss[0])
            d_accs.append(d_loss[1])
            g_losses.append(g_loss)
                
            # If at save interval => save generated image samples
            if self.epoch % save_interval == 0:
                print("Epoch {: 5d} | D loss {:4.2f} D acc {:4.2f} | G loss {:4.2f} | {:8.2%}".format(
                    self.epoch, np.mean(d_losses), np.mean(d_accs), np.mean(g_losses), (self.epoch-start_epoch)/epochs))

                # Save to history
                self.loss_discriminator.append(float(np.mean(d_losses)))
                self.acc_discriminator.append(float(np.mean(d_accs)))
                self.loss_generator.append(float(np.mean(g_losses)))
                d_losses = []
                d_accs = []
                g_losses = []
        print('-'*64)
        return {'d_loss': self.loss_discriminator,
                'd_acc': self.acc_discriminator,
                'g_loss': self.loss_generator}

# 0: 'Airplane', 1: 'Automobile', 2: 'Bird',  3: 'Cat',  4: 'Deer'
# 5: 'Dog',      6: 'Frog',       7: 'Horse', 8: 'Ship', 9: 'Truck'

dcgan = DCGAN()
dcgan.init(batch_size=64, cls=7)
history = dcgan.train(epochs=100, save_interval=25)
history = dcgan.train(epochs=200, save_interval=25)

dcgan.generator.save("DCGAN-Horses_generator.h5")
dcgan.discriminator.save("DCGAN-Horses_discriminator.h5")

with open("DCGAN-Horses.hist", 'w') as fp:
    json.dump(history, fp)
    
