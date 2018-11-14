#!/usr/bin/env python
# -*- coding: utf=8 -*-

from __future__ import print_function
import sys
import time
import matplotlib.pyplot as plt

def update_progress(msg, progress):
    barLength = 40
    status = ""
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2:.2%} {3}".format(msg, "="*(block-1) + ">" + "-"*(barLength-block), progress, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def plot_history(history):
    if 'test_loss' in history:
        print("Test loss:", history['test_loss'])
    if 'test_acc' in history:
        print("Test accuracy:", history['test_acc'])
    if 'epochs' in history:
        train_epochs = history['epochs']
        print("Final training epochs:", train_epochs)

    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], linewidth=2.0, label='train loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], linewidth=2.0, label='val loss')
    if 'test_loss' in history:
        plt.plot([train_epochs], history['test_loss'], 'rx', ms=10.0, label='test loss')
        plt.axvline(train_epochs, ls=':', linewidth=2.0)
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], linewidth=2.0, label='train acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], linewidth=2.0, label='val acc')
    if 'test_acc' in history:
        plt.plot([train_epochs], history['test_acc'], 'rx', ms=10.0, label='test acc')
        plt.axvline(train_epochs, ls=':', linewidth=2.0)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

mapping = {
           "mnist":
             ['Zero', 'One', 'Two', 'Three', 'Four',
              'Five', 'Six', 'Seven', 'Eight', 'Nine'],
           "fmnist": 
             ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
           "cifar10":
             ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
              'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
           "cifar100_coarse":
             [ 'Aquatic mammal', 'Fish', 
               'Flower', 'Food container', 
               'Fruit or vegetable', 'Household electrical device', 
               'Household furniture', 'Insect', 
               'Large carnivore', 'Large man-made outdoor thing', 
               'Large natural outdoor scene', 'Large omnivore or herbivore',
               'Medium-sized mammal', 'Non-insect invertebrate',
               'People', 'Reptile', 
               'Small mammal', 'Tree',
               'Vehicles Set 1', 'Vehicles Set 2'],
           "cifar100_fine":
             ['Apple', 'Aquarium fish', 'Baby', 'Bear', 'Beaver', 
              'Bed', 'Bee', 'Beetle', 'Bicycle', 'Bottle', 
              'Bowl', 'Boy', 'Bridge', 'Bus', 'Butterfly', 
              'Camel', 'Can', 'Castle', 'Caterpillar', 'Cattle', 
              'Chair', 'Chimpanzee', 'Clock', 'Cloud', 'Cockroach', 
              'couch', 'crab', 'crocodile', 'cups', 'dinosaur', 
              'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 
              'girl', 'hamster', 'house', 'kangaroo', 'computer keyboard',
              'lamp', 'lawn-mower', 'leopard', 'lion', 'lizard', 
              'lobster', 'man', 'maple', 'motorcycle', 'mountain', 
              'mouse', 'mushrooms', 'oak', 'oranges', 'orchids', 
              'otter', 'palm', 'pears', 'pickup truck', 'pine', 
              'plain', 'plates', 'poppies', 'porcupine', 'possum', 
              'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
              'roses', 'sea', 'seal', 'shark', 'shrew', 
              'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
              'squirrel', 'streetcar', 'sunflowers', 'sweet peppers', 'table', 
              'tank', 'telephone', 'television', 'tiger', 'tractor', 
              'train', 'trout', 'tulips', 'turtle', 'wardrobe', 
              'whale', 'willow', 'wolf', 'woman', 'worm']
          }

def decode_dataset(dataset, code):
    """Returns the string description of a Keras dataset label code
   
    # Arguments
        dataset: 'mnist', 'fmnist', 'cifar10', 'cifar100_coarse' or 'cifar100_fine'
        code: Integer code of the label.
       
    # Returns
        The corresponding description as a string
    """
    if dataset not in ['mnist', 'fmnist', 'cifar10', 'cifar100_coarse', 'cifar100_fine']:
         raise ValueError('`decode_predictions` expects '
                          'a valid dataset '
                          'Requested dataset: ' + str(dataset))
    if code not in range(len(mapping[dataset])):
         raise ValueError('Requested label code to `decode_dataset` '
                          'is out of range')
    return mapping[dataset][code]

def decode_predictions(dataset, preds, top=3):
    """Decodes the prediction of an Keras dataset model.

    # Arguments
        dataset: 'mnist', 'fmnist', 'cifar10', 'cifar100_coarse' or 'cifar100_fine'
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_number, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
                    In case of invalid requested dataset.
    """
    if dataset not in ['mnist', 'fmnist', 'cifar10', 'cifar100_coarse', 'cifar100_fine']:
        raise ValueError('`decode_predictions` expects '
                         'a valid dataset '
                         'Requested dataset: ' + str(dataset))
    
    if len(preds.shape) != 2 or preds.shape[1] != 10:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 10)). '
                         'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(i, mapping[dataset][i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

