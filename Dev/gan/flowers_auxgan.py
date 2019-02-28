#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.engine.network import Network
import keras_preprocessing

import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import glob

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 16
        self.num_classes = 17

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy','sparse_categorical_crossentropy']

        # Build and compile the discriminator
        inp_discr, out_discr, lbl_discr = self.build_discriminator()
        self.discriminator = Model(inp_discr, [out_discr, lbl_discr], name='discriminator')
        self.discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])

        # Build the frozen discriminator copy
        self.frozen_discriminator = Network(inp_discr, [out_discr, lbl_discr], name='frozen_discriminator')
        self.frozen_discriminator.trainable = False

        # Build the generator
        inp_gen, lbl_gen, out_gen = self.build_generator()
        self.generator = Model([inp_gen, lbl_gen], out_gen, name='generator')

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        optimizer = Adam(0.0002, 0.5)

        self.combined = Model([inp_gen, lbl_gen], [out_discr, lbl_discr])
        self.combined.compile(loss=losses, optimizer=optimizer)

        self.discriminator.summary()
        self.generator.summary()
        self.combined.summary()

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        labels = Input(shape=(1,))
        emb = Embedding(self.num_classes, self.latent_dim)(labels)
        flt = Flatten()(emb)
        inp = multiply([noise, flt])
        
        x = Dense(256 * 4 * 4, activation="relu")(inp)
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

        return noise, labels, img

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
        
        validity = Dense(1, activation='sigmoid')(x)
        label = Dense(self.num_classes+1, activation="softmax")(x)
        return img, validity, label

    def train(self, epochs, batch_size=128, save_interval=50):
        train_dir = '/home/sbauer/Work/Deep_Learning/Datasets/Images/17flowers/17flowers_64x64/'
        train_datagen = keras_preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x:(x/127.5) - 1.0,
            horizontal_flip=True,
            rotation_range=20.0,
            zoom_range=0.2)
        real_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(self.img_rows, self.img_cols),
            batch_size=batch_size, class_mode='categorical')

        # History arrays
        loss_discriminator = []
        acc_discriminator = []
        loss_generator = []
        d_losses = []
        d_accs = []
        g_losses = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            imgs, img_labels = next(real_generator)
    
            # Adversarial ground truths
            valid = np.ones((imgs.shape[0], 1))
            fake = np.zeros((imgs.shape[0], 1))
            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Train the discriminator (real classified as ones and generated as zeros)
            fake_labels = 10 * np.ones(img_labels.shape)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

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
        r, c = 3, self.num_classes
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        noise = np.asarray([[i/float(r*c-1)]*self.latent_dim for i in range(r*c)])
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/flowers_auxgan_{:05d}.png".format(epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    history = dcgan.train(epochs=10000, batch_size=32, save_interval=100)
    with open('flowers_auxgan.hist','w') as fp:
        json.dump(history, fp)

