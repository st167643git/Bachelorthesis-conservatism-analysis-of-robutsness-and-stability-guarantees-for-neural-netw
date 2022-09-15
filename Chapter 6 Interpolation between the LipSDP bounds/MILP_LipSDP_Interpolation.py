# formulate and solve MILP for interpolation between LipSDP Neuron and Layer.

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matlab
import matlab.engine
import pandas as pd
import json
import time
from operator import itemgetter

eng = matlab.engine.start_matlab()

def neuron_norm_quotient(W, W_prior, j_neuron):
    a_j_neuron = np.log(np.linalg.norm(W_prior[j_neuron, :], 2)/np.linalg.norm(W[:, j_neuron], 2))
    return a_j_neuron

def interpol_MILP(W, W_prior, n_DOF, time_limit):
    n_neurons = len(W_prior)

    a_neuron = {j: neuron_norm_quotient(W, W_prior, j) for j in range(n_neurons)}
    print('a=', a_neuron)

    # determine  minimum and maximum neuron norm quotient
    L = min(a_neuron.values())
    U = max(a_neuron.values())

    model = gp.Model('Interpolation_LipSDP')

    # set number of threads for comparability of the computation time
    # model.Params.Threads = 1

    # Add decision variables
    # binary variables accessed as x[0,0], x[0,1] etc., determine if a neuron is part of a subset
    x = model.addVars(n_DOF, n_neurons, vtype=GRB.BINARY)

    # continuous variables y_k for each subset
    y = model.addVars(n_DOF,  lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

    # continuous variables z_k for each subset
    z = model.addVars(n_DOF, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

    # variable c for the may difference of elements in the subset
    c = model.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS)

    # Add constraints
    # 1. one neuron belongs to exactly one subset
    cons1 = model.addConstrs(x.sum('*', j) == 1 for j in range(n_neurons))

    # 2. use all DOFs, no empty subsets
    cons2 = model.addConstrs(x.sum(k, '*') >= 1 for k in range(n_DOF))

    # 3. constraints for y_k, greater than the maximum element in the subset k
    cons3 = model.addConstrs(y[k] >= (a_neuron[j] * x[k, j] + L * (1 - x[k, j])) for k in range(n_DOF) for j in range(n_neurons))

    # 4. constraints for z_k, smaller than the maximum element in the subset k
    cons4 = model.addConstrs(z[k] <= (a_neuron[j] * x[k, j] + U * (1 - x[k, j])) for k in range(n_DOF) for j in range(n_neurons))

    # constraint for c, objective
    cons5 = model.addConstrs(c >= (y[k]-z[k]) for k in range(n_DOF))

    # set c as objective
    model.setObjective(c, GRB.MINIMIZE)

    # set time limit
    model.setParam('TimeLimit', time_limit)

    # optimize
    model.optimize()

    # get runtime
    runtime = model.Runtime
    print(runtime)

    # get solution
    solx = model.getAttr('x', x)

    # convert to matrix of zeros and ones
    xmat = np.zeros((n_DOF, n_neurons))
    for i in range(n_DOF):
        for j in range(n_neurons):
            xmat[i, j] = solx[i, j]

    return xmat, runtime

def LipSDP_Interpolate_MILP(weights, n_DOF, time_limit_overhead_total):

    # net dimensions from weight tensor
    net_dim = []
    net_dim.append(len(weights[0][:, 1]))
    for W in weights:
        net_dim.append(len(W))
    n_hidden_layers = len(net_dim) - 2
    net_dim = matlab.int64([net_dim])

    time_limit = time_limit_overhead_total/ n_hidden_layers
    x_list = []
    time_total_solver = 0
    start_time_overhead = time.time()
    for hidden_layer in range(n_hidden_layers):
        W_prior = weights[hidden_layer]
        W = weights[hidden_layer + 1]
        x, runtime = interpol_MILP(W, W_prior, n_DOF, time_limit)
        time_total_solver = time_total_solver + runtime
        x_list.append(x)
    time_overhead_MILP = time.time() - start_time_overhead

    weights_dict = {}
    for i in range(len(weights)):
        weights_dict.update({
            'W{:d}'.format(i): np.array(weights, dtype=np.object)[i].tolist()})
    with open('weights.json', 'w') as f1:  # writing JSON object
        json.dump(weights_dict, f1)

    x_dict = {}
    for i in range(len(x_list)):
        x_dict.update({
            'X{:d}'.format(i): x_list[i].tolist()})
    with open('X_dict.json', 'w') as f2:  # writing JSON object
        json.dump(x_dict, f2)

    L_LipSDP_Interpolate, time_solver_SDP, time_overhead_SDP = eng.LipSDP_Interpolate_Matlab(net_dim, nargout=3)

    return L_LipSDP_Interpolate, time_total_solver, time_overhead_MILP, time_solver_SDP, time_overhead_SDP

def interpol_Heuristik(W, W_prior, n_DOF):
    n_neurons = len(W_prior)

    # list of norm quotients belonging to the neurons
    a_neuron = [(j, neuron_norm_quotient(W, W_prior, j)) for j in range(n_neurons)]


    # sort the list w.r.t the norm quotients
    a_neuron = sorted(a_neuron, key=itemgetter(1))

    # compute number of neurons per DOF, distribute as equally as possible
    small_int = np.floor_divide(n_neurons, n_DOF)
    n_big_int = np.mod(n_neurons, n_DOF)
    n_small_int = n_DOF - n_big_int
    big_int = small_int + 1

    # choose n_DOF times the smaller int and n_bigint times the bigger int, to distribute the neurons as equally as possible
    x = []
    already_selected = 0
    for dof_small_int in range(n_small_int):
        x_row = np.zeros(n_neurons)
        for entry in range(small_int):
            entry = entry + already_selected
            j_neuron = a_neuron[entry][0]
            x_row[j_neuron] = 1
        already_selected = already_selected + small_int
        x.append(x_row)

    for dof_big_int in range(n_big_int):
        x_row = np.zeros(n_neurons)
        for entry in range(big_int):
            entry = entry + already_selected
            j_neuron = a_neuron[entry][0]
            x_row[j_neuron] = 1
        already_selected = already_selected + big_int
        x.append(x_row)

    xmat = np.stack(x, axis=0)

    return xmat

def LipSDP_Interpolate_Heuristik(weights, n_DOF):

    # net dimensions from weight tensor
    net_dim = []
    net_dim.append(len(weights[0][:, 1]))
    for W in weights:
        net_dim.append(len(W))
    n_hidden_layers = len(net_dim) - 2
    net_dim = matlab.int64([net_dim])

    start_time_overhead = time.time()

    x_list = []
    for hidden_layer in range(n_hidden_layers):
        W_prior = weights[hidden_layer]
        W = weights[hidden_layer + 1]
        x = interpol_Heuristik(W, W_prior, n_DOF)
        x_list.append(x)

    time_overhead_Heuristik = time.time() - start_time_overhead

    weights_dict = {}
    for i in range(len(weights)):
        weights_dict.update({
            'W{:d}'.format(i): np.array(weights, dtype=np.object)[i].tolist()})
    with open('weights.json', 'w') as f1:  # writing JSON object
        json.dump(weights_dict, f1)

    x_dict = {}
    for i in range(len(x_list)):
        x_dict.update({
            'X{:d}'.format(i): x_list[i].tolist()})
    with open('X_dict.json', 'w') as f2:  # writing JSON object
        json.dump(x_dict, f2)

    L_LipSDP_Interpolate, time_solver, time_overhead_SDP = eng.LipSDP_Interpolate_Matlab(net_dim, nargout=3)

    return L_LipSDP_Interpolate, time_overhead_Heuristik, time_solver, time_overhead_SDP

# random interpolation for benchmarking
def interpol_Random(W, W_prior, n_DOF):
    n_neurons = len(W_prior)

    # choose dof index randomly
    DOFs_neurons = np.random.randint(0, n_DOF, size=n_neurons)

    # make sure each dof is used
    for dof in range(n_DOF):
        if dof not in DOFs_neurons:
            while True:
                index = np.random.randint(0, n_neurons)
                # make sure to not overwrite the only index of an DOF
                if np.count_nonzero(DOFs_neurons == DOFs_neurons[index]) > 1:
                    break
            DOFs_neurons[index] = dof

    x = []
    for dof in range(n_DOF):
        x_row = np.zeros(n_neurons)
        indixes = np.where(DOFs_neurons == dof)
        for index in indixes:
            x_row[index] = 1
        x.append(x_row)

    xmat = np.stack(x, axis=0)

    return xmat

# LipSDP with random assignment of neurons to DOFs for benchmarking
def LipSDP_Interpolate_Random(weights, n_DOF):

    # net dimensions from weight tensor
    net_dim = []
    net_dim.append(len(weights[0][:, 1]))
    for W in weights:
        net_dim.append(len(W))
    n_hidden_layers = len(net_dim) - 2
    net_dim = matlab.int64([net_dim])

    x_list = []
    for hidden_layer in range(n_hidden_layers):
        W_prior = weights[hidden_layer]
        W = weights[hidden_layer + 1]
        x = interpol_Random(W, W_prior, n_DOF)
        x_list.append(x)

    weights_dict = {}
    for i in range(len(weights)):
        weights_dict.update({
            'W{:d}'.format(i): np.array(weights, dtype=np.object)[i].tolist()})
    with open('weights.json', 'w') as f1:  # writing JSON object
        json.dump(weights_dict, f1)

    x_dict = {}
    for i in range(len(x_list)):
        x_dict.update({
            'X{:d}'.format(i): x_list[i].tolist()})
    with open('X_dict.json', 'w') as f2:  # writing JSON object
        json.dump(x_dict, f2)

    L_LipSDP_Interpolate, time_solver_SDP, time_overhead_SDP = eng.LipSDP_Interpolate_Matlab(net_dim, nargout=3)

    return L_LipSDP_Interpolate, time_solver_SDP, time_overhead_SDP

