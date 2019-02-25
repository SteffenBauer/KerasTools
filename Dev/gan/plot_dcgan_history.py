#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("history")
args = parser.parse_args()

with open(args.history, 'rb') as fp:
    history = json.load(fp)

epochs = range(1, len(history['g_loss']) + 1)

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)

plt.title(args.history)
plt.plot(epochs, history['d_loss'], label='Discriminator loss')
plt.plot(epochs, history['g_loss'], label='Generator loss')

#axes = plt.gca()
#axes.set_ylim([0.0,2.0])

plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history['d_acc'], label='Discriminator accuracy')
axes = plt.gca()
axes.set_ylim([0.0,1.0])

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.show()

