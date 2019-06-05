#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import keras
import KerasTools as KT

import matplotlib.pyplot as plt
import numpy as np
import json

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 16

        if not os.path.exists("./images"):
            os.makedirs("./images")

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        inp_discriminator, out_discriminator = self.build_discriminator()
        self.discriminator = keras.models.Model(inp_discriminator, out_discriminator, name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the frozen discriminator copy
        self.frozen_discriminator = keras.engine.network.Network(inp_discriminator, out_discriminator, name='frozen_discriminator')
        self.frozen_discriminator.trainable = False

        # Build the generator
        inp_generator, out_generator = self.build_generator()
        self.generator = keras.models.Model(inp_generator, out_generator, name='generator')

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        self.combined = keras.models.Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.frozen_discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.discriminator.summary()
        self.generator.summary()
        self.combined.summary()

    def build_generator(self):

        noise = keras.layers.Input(shape=(self.latent_dim,))

        x = keras.layers.Dense(128 * 7 * 7, activation="relu")(noise)
        x = keras.layers.Reshape((7, 7, 128))(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(128, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(self.channels, kernel_size=3, padding="same")(x)
        img = keras.layers.Activation("tanh")(x)

        return noise, img

    def build_discriminator(self):

        img = keras.layers.Input(shape=self.img_shape)

        x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Flatten()(x)
        validity = keras.layers.Dense(1, activation='sigmoid')(x)

        return img, validity

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, Y_train), (_, _) = keras.datasets.mnist.load_data()

        X_train = X_train[Y_train.flatten() == 8]

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # History arrays
        loss_discriminator = []
        acc_discriminator = []
        loss_generator = []
        d_losses = []
        d_accs = []
        g_losses = []

        for epoch in range(epochs+1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Record loss/acc history
            d_losses.append(d_loss[0])
            d_accs.append(d_loss[1])
            g_losses.append(g_loss)
            KT.KerasTools.update_progress("Epoch {: 5d} | D loss {:2.2f} D acc {:2.2f} | G loss {:2.2f}".format(
                epoch, np.mean(d_losses), np.mean(d_accs), np.mean(g_losses)), (((epoch%save_interval)+1)/save_interval))

            # If at save interval => save generated image samples
            if epoch == 0: 
                self.save_imgs(epoch)
            if (epoch % save_interval)+1 == save_interval:
                self.save_imgs(epoch)
                if epoch != 0:
                    print()
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
        r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        noise = np.asarray([[i/24.0]*self.latent_dim for i in range(25)])
        
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_{:05d}.png".format(epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    history = dcgan.train(epochs=200, batch_size=32, save_interval=100)
    with open('mnist_gan.hist','w') as fp:
        json.dump(history, fp)

