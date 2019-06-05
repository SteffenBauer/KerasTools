#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import keras
import keras_preprocessing

height, width, channels = 64, 64, 3
preproc_mode = 'tf'

train_dir = '/home/sbauer/Work/Deep_Learning/Datasets/Images/17flowers/17flowers_64x64/'

def build_network(shape):

    inp = keras.layers.Input(shape=shape)

    x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(inp)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(17, activation='softmax')(x)

    model = keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(optimizer='RMSProp',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

train_datagen = keras_preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x:x/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(height, width), batch_size=24, class_mode='categorical', subset='training')
val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(height, width), batch_size=17, class_mode='categorical', subset='validation')

model = build_network(shape=(height, width, channels))
model.summary()

history = model.fit_generator(
    train_generator, validation_data=val_generator,
    epochs=100, use_multiprocessing=True, workers=4)

with open('./17flowers.hist', 'w') as fp:
    json.dump(history.history, fp)

