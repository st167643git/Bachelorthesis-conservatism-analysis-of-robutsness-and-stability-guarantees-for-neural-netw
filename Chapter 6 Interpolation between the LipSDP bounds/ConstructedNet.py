import numpy as np

# construcs a Net with weights matrices with half of the neurons having large norm quotients, half having small ones
# n_neurons_per_layer needs to be
def constructedNet1(n_layers, n_neurons_per_layer,factor):

    # construct weight matrices
    weights = []

    # construct the first weight matrix, is the weight matrix for every second layer
    W_1 = 10**(-factor)*np.ones((n_neurons_per_layer, n_neurons_per_layer))
    for i in range(int(n_neurons_per_layer/2)):
        for j in range(int(n_neurons_per_layer/2)):
            W_1[i, j] = 10**factor

    # scale the problem to avoid issues with Numeriks, scaling does not change the realtive error between lipSDP layer and neuron
    W_1 = W_1/np.max(W_1)

    # second weights matrix, for any other layer
    W_2 = 10**(-factor)*np.ones((n_neurons_per_layer, n_neurons_per_layer))
    for i in range(int(n_neurons_per_layer / 2)):
        index_1 = i + int(n_neurons_per_layer / 2)
        for j in range(int(n_neurons_per_layer / 2)):
            index_2 = j + int(n_neurons_per_layer / 2)
            W_2[index_1, index_2] = 10 ** factor

    # scale the problem to avoid issues with Numeriks, scaling does not change the realtive error between lipSDP layer and neuron
    W_2 = W_2/np.max(W_2)

    # switch between both matrices
    switch = 0
    for layer in range(n_layers + 1):
        if switch == 0:
            weights.append(W_1)
            switch = 1
        else:
            weights.append(W_2)
            switch = 0

    return weights

def constructedNet2(n_layers, n_neurons_per_layer,factor):

    # construct weight matrices
    weights = []

    # construct the first weight matrix, is the weight matrix for every second layer
    W_1 = np.ones((n_neurons_per_layer, n_neurons_per_layer))
    subblock = 1
    while subblock <= n_neurons_per_layer - 1:
        for i in range(int(n_neurons_per_layer-subblock)):
            index_1 = i + subblock
            for j in range(int(n_neurons_per_layer-subblock)):
                index_2 = j + subblock
                W_1[index_1, index_2] = (2**factor)*W_1[index_1, index_2]
        subblock = subblock + 1
    # rescaling, does not change the relative error between lipSDP Neuron and Layer
    W_1 = W_1/np.max(W_1)

    # second weights matrix, for any other layer
    W_2 = np.ones((n_neurons_per_layer, n_neurons_per_layer))
    subblock = 1
    while subblock <= n_neurons_per_layer - 1:
        for i in range(int(n_neurons_per_layer - subblock)):
            for j in range(int(n_neurons_per_layer - subblock)):
                W_2[i, j] = (2 ** factor) * W_2[i, j]
        subblock = subblock + 1
    # rescaling, does not change the relative error between lipSDP Neuron and Layer
    W_2 = W_2/np.max(W_2)

    # switch between both matrices
    switch = 0
    for layer in range(n_layers + 1):
        if switch == 0:
            weights.append(W_1)
            switch = 1
        else:
            weights.append(W_2)
            switch = 0

    return weights
