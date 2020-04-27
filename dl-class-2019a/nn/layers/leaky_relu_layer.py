import numpy as np
from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.prev_data = None
        self.slope = slope

    def forward(self, data):
        # TODO
        self.prev_data = data
        return np.where(data > 0, data, self.slope * data)

    def backward(self, previous_partial_gradient):
        # TODO
        return previous_partial_gradient * np.where(self.prev_data > 0 , 1, self.slope)
