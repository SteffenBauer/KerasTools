#!/usr/bin/env python3

import time
import os

import keras
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Prepare training & test data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype(keras.backend.floatx()) / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype(keras.backend.floatx()) / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Build the network
def network_basic():
    inp = keras.layers.Input(shape = (28, 28, 1), name='Input')
    x = keras.layers.Conv2D(20, (3, 3), activation='relu', name='Conv_1')(inp)
    x = keras.layers.MaxPooling2D((2, 2), name='Pool_1')(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu', name='Conv_2')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='Pool_2')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(500, activation='relu')(x)
    out = keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    network = keras.models.Model(inputs=inp, outputs=out)
    return network

start_time = time.time()
network = network_basic()
end_time = time.time()
print("Build time", end_time - start_time)

start_time = time.time()
network.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
end_time = time.time()
print("Compilation time", end_time - start_time)

network.summary()

# Train and test the network
start_time = time.time()
history = network.fit(train_images, train_labels, epochs=10, batch_size=64)
end_time = time.time()
print("Training time", end_time - start_time)

start_time = time.time()
test_loss, test_acc = network.evaluate(test_images, test_labels)
end_time = time.time()
print()
print("Test loss", test_loss)
print("Test accuracy", test_acc)
print("Test inference time", end_time - start_time)

