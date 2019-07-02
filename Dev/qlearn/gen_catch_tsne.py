#!/usr/bin/env python3

import pickle
import keras
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

with open('./train_data/catch_mem.pkl', 'rb') as fp:
    memory = pickle.load(fp)
states = []
for e,h in memory:
    S,_,_,_,_ = e
    states.append(S)

for i in range(51):
    model = keras.models.load_model('./train_data/catch_{:02d}.h5'.format(i))
    activation_model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    activation_model.summary()

    result = activation_model.predict([states])
    actions = np.argmax(model.predict([states]), axis=-1)
    print(i, result.shape, actions.shape)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(result)

    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=actions, cmap='prism', s=2)
    plt.savefig('./train_data/catch_tsne_{:02d}.png'.format(i))
    plt.clf()

