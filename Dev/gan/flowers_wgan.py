#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.engine.network import Network
import keras_preprocessing
import keras.backend as K

import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import glob
import cv2 

class WGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 16

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        inp_discriminator, out_discriminator = self.build_discriminator()
        self.discriminator = Model(inp_discriminator, out_discriminator, name='discriminator')
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        # Build the frozen discriminator copy
        self.frozen_discriminator = Network(inp_discriminator, out_discriminator, name='frozen_discriminator')
        self.frozen_discriminator.trainable = False

        # Build the generator
        inp_generator, out_generator = self.build_generator()
        self.generator = Model(inp_generator, out_generator, name='generator')

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        optimizer = Adam(0.0002, 0.5)

        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.frozen_discriminator)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        self.discriminator.summary()
        self.generator.summary()
        self.combined.summary()

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))

        x = Dense(256 * 4 * 4, activation="relu")(noise)
        x = Reshape((4, 4, 256))(x)
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        x = UpSampling2D()(x)
        x = Conv2D(32, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        x = UpSampling2D()(x)
        x = Conv2D(16, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        x = Conv2D(self.channels, kernel_size=3, padding="same")(x)
        img = Activation("tanh")(x)

        return noise, img

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        validity = Dense(1)(x)

        return img, validity

    def train(self, epochs, batch_size=128, save_interval=50):
        train_dir = '/home/sbauer/Work/Deep_Learning/Datasets/Images/17flowers/17flowers_64x64/'
        train_datagen = keras_preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x:(x/127.5) - 1.0,
            horizontal_flip=True,
            rotation_range=20.0,
            zoom_range=0.2)
        real_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(self.img_rows, self.img_cols), 
            batch_size=batch_size, class_mode='categorical', classes=['dandelion'])

        # History arrays
        loss_discriminator = []
        acc_discriminator = []
        loss_generator = []
        d_losses = []
        d_accs = []
        g_losses = []

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                imgs = next(real_generator)[0]
                valid = -np.ones((imgs.shape[0], 1))
                fake = np.ones((imgs.shape[0], 1))
            
                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            d_losses.append(d_loss[0])
            d_accs.append(d_loss[1])
            g_losses.append(g_loss)
            
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                # Save to history
                loss_discriminator.append(float(np.mean(d_losses)))
                acc_discriminator.append(float(np.mean(d_accs)))
                loss_generator.append(float(np.mean(g_losses)))
                d_losses = []
                d_accs = []
                g_losses = []
        
        return {'d_loss': loss_discriminator,
                'd_acc': acc_discriminator,
                'g_loss': loss_generator}

    def save_imgs(self, epoch):
        r, c = 3, 3
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        noise = np.asarray([[i/float(r*c-1)]*self.latent_dim for i in range(r*c)])
        
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/flowers_wgan_{:05d}.png".format(epoch))
        plt.close()


if __name__ == '__main__':
    wgan = WGAN()
    history = wgan.train(epochs=10000, batch_size=32, save_interval=100)
    with open('flowers_wgan.hist','w') as fp:
        json.dump(history, fp)

