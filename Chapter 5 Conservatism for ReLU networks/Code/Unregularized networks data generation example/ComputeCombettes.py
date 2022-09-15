# function for computing the upper bound by combettes
# input  weights as needed for LipBaB, hidden_layer_dim list af dimensions of hidden layers (exclude input/output layers)

import numpy as np
import numpy.linalg as linalg
import itertools

def Theta_m(weights,hidden_layer_dim):

    matrix_list = weights

    n_hidden_layers = len(hidden_layer_dim)
    combinations_array = []

    n_hidden_neurons = np.sum(hidden_layer_dim)
    lst = list(map(list, itertools.product([0, 1], repeat=n_hidden_neurons)))
    combinations_array.append(lst)

    theta_m = 0
    for pattern in lst:
        last_value = 0
        W_ges = np.array(matrix_list[0+1])
        #print(W_ges)
        for n_layer in range(n_hidden_layers):
            Lambda = np.diag(pattern[last_value:last_value+hidden_layer_dim[n_layer]])
            #print(Lambda)
            last_value = last_value+hidden_layer_dim[n_layer]
            W_ges = np.matmul(Lambda, W_ges)
            #print(W_ges)
            W_next = np.array(matrix_list[n_layer+2])
            #print(W_next)
            W_ges = np.matmul(W_next, W_ges)
        Norm_linear_subregion = linalg.norm(W_ges,2)
        #print(Norm_linear_subregion)
        if Norm_linear_subregion > theta_m:
            theta_m = Norm_linear_subregion
    return theta_m