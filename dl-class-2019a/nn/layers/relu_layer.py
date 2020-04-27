import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.prev_data = None

    def forward(self, data):
        # TODO
        self.prev_data = data
        return np.maximum(0, data)

    def backward(self, previous_partial_gradient):
        # TODO
        return previous_partial_gradient * (self.prev_data > 0)


class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        return np.maximum(0, data)

    def forward(self, data):
        # TODO
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        return grad * (data > 0)

    def backward(self, previous_partial_gradient):
        # TODO
        return self.backward_numba(self.data, previous_partial_gradient)
