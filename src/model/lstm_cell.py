
"""

FROM: Coursera

"""

import numpy as np
from src.utils.activations import *
from src.utils.cuda import *

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray

import skcuda.linalg as linalg

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.zeros(((n_a + n_x), m))
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure
    ft = sigmoid(np.matmul(Wf, concat) + bf)
    it = sigmoid(np.matmul(Wi, concat) + bi)
    cct = np.tanh(np.matmul(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.matmul(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = softmax(np.matmul(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_cell_forward_gpu(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell using PyCUDA

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """
    linalg.init()

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.zeros(((n_a + n_x), m))
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    concat = pycuda.gpuarray.to_gpu(concat)
    c_prev = pycuda.gpuarray.to_gpu(c_prev)

    # Compute values for ft, it, cct, c_next, ot, a_next
    ft = sigmoid_gpu(add_bias_gpu(linalg.dot(Wf, concat), bf))
    it = sigmoid_gpu(add_bias_gpu(linalg.dot(Wi, concat), bi))
    cct = tanh_gpu(add_bias_gpu(linalg.dot(Wc, concat), bc))
    c_next = elem_mul_gpu(ft, c_prev) + elem_mul_gpu(it, cct)
    ot = sigmoid_gpu(add_bias_gpu(linalg.dot(Wo, concat), bo))
    a_next = elem_mul_gpu(ot, tanh_gpu(c_next))

    # Compute prediction of the LSTM cell
    yt_pred = softmax_gpu(add_bias_gpu(linalg.dot(Wy, a_next), by))

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # Compute gates related derivatives
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot *(1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute parameters related derivatives.
    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)
    dbf = np.sum(dft, axis=1 ,keepdims = True)
    dbi = np.sum(dit, axis=1, keepdims = True)
    dbc = np.sum(dcct, axis=1,  keepdims = True)
    dbo = np.sum(dot, axis=1, keepdims = True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input.
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)
    
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients


def lstm_cell_backward_gpu(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step) on gpu.

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """
    linalg.init()

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives
    dot = elem_mul_gpu(elem_mul_gpu(da_next, tanh_gpu(c_next)), elem_mul_gpu(ot, from_one_gpu(ot)))
    dcct = elem_mul_gpu(dc_next, it) + elem_mul_gpu(elem_mul_gpu(ot, from_one_gpu(elem_mul_gpu(tanh_gpu(c_next),tanh_gpu(c_next)))),
                                               elem_mul_gpu(it, da_next), from_one_gpu(elem_mul_gpu(cct,cct)))
    dit = elem_mul_gpu(dc_next, cct) + elem_mul_gpu(elem_mul_gpu(ot,from_one_gpu(elem_mul_gpu(tanh_gpu(c_next),tanh_gpu(c_next)))),
                                                  elem_mul_gpu(elem_mul_gpu(cct, da_next),
                                                             elem_mul_gpu(it, from_one_gpu(it))))
    dft = elem_mul_gpu(elem_mul_gpu(dc_next, c_prev) +
                     elem_mul_gpu(ot,
                                elem_mul_gpu(from_one_gpu(elem_mul_gpu(tanh_gpu(c_next),tanh_gpu(c_next))), elem_mul_gpu(c_prev, da_next)),
                     elem_mul_gpu(ft, from_one_gpu(ft))))
    concat = np.concatenate((a_prev, xt), axis=0)
    concat = pycuda.gpuarray.to_gpu(concat)

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    dWf = linalg.dot(dft, concat.T)
    dWi = linalg.dot(dit, concat.T)
    dWc = linalg.dot(dcct, concat.T)
    dWo = linalg.dot(dot, concat.T)
    dbf = pycuda.gpuarray.sum(dft)
    dbi = pycuda.gpuarray.sum(dit)
    dbc = pycuda.gpuarray.sum(dcct)
    dbo = pycuda.gpuarray.sum(dot)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input.
    da_prev = linalg.dot(parameters['Wf'][:, :n_a].T, dft) + linalg.dot(parameters['Wi'][:, :n_a].T,
                                                                          dit) + linalg.dot(
        parameters['Wc'][:, :n_a].T, dcct) + linalg.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = linalg.dot(dc_next, ft) + linalg.dot(elem_mul_gpu(ot, from_one_gpu(linalg.dot(tanh_gpu(c_next),tanh_gpu(c_next)))),
                                                   linalg.dot(ft, da_next))
    dxt = linalg.dot(parameters['Wf'][:, n_a:].T, dft) + linalg.dot(parameters['Wi'][:, n_a:].T, dit) + linalg.dot(
        parameters['Wc'][:, n_a:].T, dcct) + linalg.dot(parameters['Wo'][:, n_a:].T, dot)

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


def cell_to_gpu(parameters: dict):
    """
    copy cell weights to GPU
    :param parameters: dictionary of cell weights
    :return: gpu_params: parameters transferred as PyCuda GPUArray
    """
    gpu_params = {}
    for parameter in parameters.items():
        gpu_params[parameter[0]] = pycuda.gpuarray.to_gpu(parameter[1])

    return gpu_params


def cell_from_gpu(gpu_parameters: dict):
    """
    copy cell weights from GPU
    :param gpu_parameters: dictionary of cell weights with weights being PyCuda GPUArrays
    :return: parameters: parameters transferred from PyCuda GPUArray to numpy arrays
    """
    parameters = {}
    for gpu_param in gpu_parameters.items():
        parameter = gpu_param[1].get()
        parameters[gpu_param[0]] = parameter

    return parameters
