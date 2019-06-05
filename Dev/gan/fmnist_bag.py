#!/usr/bin/env python

import numpy as np
import json
import time
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import callbacks

from collections import Counter

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Prepare training & test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype(K.floatx()) / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype(K.floatx()) / 255

train_labels = np.asarray([1 if l == 8 else 0 for l in train_labels]).astype('float32')
test_labels = np.asarray([1 if l == 8 else 0 for l in test_labels]).astype('float32')

counts = Counter(train_labels)
weights = {k: float(len(train_labels)) / float(2 * v) for k,v in counts.items()}
print weights

# Build the network
def network_basic():
    inp = layers.Input(shape = (28, 28, 1), name='Input')
    x = layers.Conv2D(32, (3, 3), activation='relu', name='Conv_1')(inp)
    x = layers.MaxPooling2D((2, 2), name='Pool_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', name='Conv_2')(x)
    x = layers.MaxPooling2D((2, 2), name='Pool_2')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', name='Conv_3')(x)
    x = layers.GlobalAveragePooling2D(name='Global_Avg_Pool')(x)
    out = layers.Dense(1, activation='sigmoid', name='classifier')(x)
    network = models.Model(inputs=inp, outputs=out)
    return network

network = network_basic()
network.compile(optimizer=optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy', precision, recall])
network.summary()

# Train and test the network
history = network.fit(train_images, train_labels, batch_size=32, epochs=10, validation_split=0.1, class_weight=weights)

start_time = time.time()
test_loss, test_acc, test_prec, test_rec = network.evaluate(test_images, test_labels)
end_time = time.time()
print
print "Test loss", test_loss
print "Test accuracy", test_acc
print "Test precision", test_prec
print "Test recall", test_rec
print "Test inference time", end_time - start_time
history.history['test_loss'] = test_loss
history.history['test_acc'] = test_acc
history.history['test_prec'] = test_prec
history.history['test_rec'] = test_rec

# Save network and training history
#network.save("./fmnist_trained_{}.h5".format(K.floatx()))
#with open('./fmnist_bag_{}.hist'.format('basic'), 'wb') as fp:
#    json.dump(history.history, fp)

preds = network.predict(test_images)
true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0
for i in range(len(test_labels)):
    if test_labels[i] > 0.5 and preds[i] > 0.5:
        true_pos += 1
    elif test_labels[i] > 0.5 and preds[i] <= 0.5:
        false_neg += 1
    elif test_labels[i] <= 0.5 and preds[i] > 0.5:
        false_pos += 1
    else:
        true_neg += 1
print true_pos, true_neg
print false_pos, false_neg
print "Precision: ", float(true_pos) / float(true_pos + false_pos)
print "Recall: ", float(true_pos) / float(true_pos + false_neg)
print "Specificity: ", float(true_neg) / float(true_neg + false_pos)
print "Miss rate: ", float(true_neg) / float(true_neg + true_pos)

