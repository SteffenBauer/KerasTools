#!/usr/bin/env python

import json
import matplotlib.pyplot as plt

histories = ['sgd-mom', 'adabound']
history = dict()
for h in histories:
    with open("cifar10_densenet_{}.hist".format(h), 'r') as fp:
        history[h] = json.load(fp)

epochs = list(range(1, len(history['sgd-mom']['loss']) + 1))

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)

for h in histories:
    plt.plot(epochs, history[h]['val_loss'], label=h)

plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)

for h in histories:
    plt.plot(epochs, history[h]['val_acc'], label=h)

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.show()

