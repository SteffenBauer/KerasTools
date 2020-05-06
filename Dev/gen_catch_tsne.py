#!/usr/bin/env python3

import pickle
import keras
import numpy as np
from sklearn import manifold, decomposition, neighbors
import matplotlib.pyplot as plt

with open('./train_data/catch_mem.pkl', 'rb') as fp:
    memory = pickle.load(fp)
states = []
for e,h in memory:
    S,_,_,_,_ = e
    states.append(S)
#states = states[:100]

for i in range(31):
    model = keras.models.load_model('./train_data/catch_{:02d}.h5'.format(i))
    activation_model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    result = activation_model.predict([states])
    actions = np.argmax(model.predict([states]), axis=-1)
    classifier_params = model.layers[-1].get_weights()
    weights = classifier_params[0]
    biases = classifier_params[1]
    weighted_result = []
    for r,a in zip(result, actions):
        m = np.multiply(r, weights[:, a]) + biases[a]
        weighted_result.append(m)

    print(i, result.shape, actions.shape)

    #x_min, x_max = np.amin(weighted_result), np.amax(weighted_result)
    #scaled_result = (weighted_result - x_min) / (x_max - x_min)

    X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(weighted_result)
#    X_lle, _ = manifold.locally_linear_embedding(weighted_result, n_components=2, n_neighbors=30)
#    X_pca = decomposition.PCA(n_components=2).fit_transform(scaled_result)
#    X_iso = manifold.Isomap(30, n_components=2).fit_transform(weighted_result)
#    X_nca = neighbors.NeighborhoodComponentsAnalysis(n_components=2, random_state=0).fit_transform(weighted_result, actions)
#    X_ltsa = manifold.LocallyLinearEmbedding(30, 2, eigen_solver='auto', method='ltsa').fit_transform(scaled_result)

    X_result = X_tsne
    
    plt.scatter(X_result[:,0], X_result[:,1], c=actions, cmap='prism', s=2)
    plt.savefig('./train_data/catch_tsne_{:02d}.png'.format(i))
    plt.clf()

