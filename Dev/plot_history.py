#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("history")
args = parser.parse_args()

with open(args.history, 'r') as fp:
    history = json.load(fp)

epochs = list(range(1, len(history['loss']) + 1))
train_epochs = history.get('epochs', epochs[-1])

if 'acc' in history:
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)

plt.title(args.history)
plt.plot(epochs, history['loss'], label='train loss')
if 'val_loss' in history:
    plt.plot(epochs, history['val_loss'], label='val loss')
if 'test_loss' in history:
    plt.plot([train_epochs], history['test_loss'], 'rx', ms=5.0, label='test loss')
    plt.axvline(train_epochs, ls=':')
if 'baseline' in history:
    plt.axhline(history['baseline'], ls=':', linewidth=2.0, color='red', label='baseline')

plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

if 'acc' in history:
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], label='train acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], label='val acc')
    if 'test_acc' in history:
        plt.plot([train_epochs], history['test_acc'], 'rx', ms=5.0, label='test acc')
        plt.axvline(train_epochs, ls=':')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

plt.show()

if 'test_loss' in history:
    print("Test loss:", history['test_loss'])
if 'test_acc' in history:
    print("Test accuracy:", history['test_acc'])

