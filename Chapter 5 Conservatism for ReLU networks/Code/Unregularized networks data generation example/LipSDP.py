# functions for calling matlab and solving the SDP of LipSDP

import torch
import matlab.engine
import numpy as np
import json

eng = matlab.engine.start_matlab()


def lipSDP(weights, net_dims, decisionstring):
    x = matlab.int64([net_dims])

    parameters = {}
    weights_dict = {}
    biases_dict = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        weights_dict.update({
            'W{:d}'.format(i): np.array(weights, dtype=np.object)[i].tolist() })

    with open('weights.json', 'w') as f:  # writing JSON object
        json.dump(weights_dict, f)
    with open('biases.json', 'w') as f:  # writing JSON object
        json.dump(biases_dict, f)
    Lip = eng.LipSDP(parameters, x, decisionstring)
    return Lip