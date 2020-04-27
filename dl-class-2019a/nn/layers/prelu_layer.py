import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.prev_data = None
        self.size = size
        self.slope = Parameter(np.full(size, initial_slope))

    def forward(self, data):
        # TODO
        self.prev_data = data
        # Channel-Shared Coeff
        if self.size == 1:
            return np.where(data > 0, data, self.slope.data * data)
        # Channel-Wise Coeff
        elif self.size > 1:
            return np.where(data > 0, data, self.slope.data[:,np.newaxis] * data)
        else:
            raise("size input should be an integer >= 1")

    def backward(self, previous_partial_gradient):
        # TODO 
        # Ref: https://arxiv.org/pdf/1502.01852.pdf

        # Channel-Shared Coeff
        activation_gradient = np.where(self.prev_data > 0, 0, self.prev_data)
        if self.size == 1:
            print(self.slope.data)
            self.slope.grad = np.sum(activation_gradient * previous_partial_gradient)
            return previous_partial_gradient * np.where(self.prev_data > 0, 1, self.slope.data)
        # Channel-Wise Coeff
        elif self.size > 1:
            #print(np.where(self.prev_data > 0, 0, self.prev_data) * previous_partial_gradient)
            raw_slope_grad = np.moveaxis(np.where(self.prev_data > 0, 0, self.prev_data * previous_partial_gradient), 0, 1)
            self.slope.grad = np.sum(np.reshape(raw_slope_grad, (self.size, -1)), axis=-1)
            return previous_partial_gradient * np.where(self.prev_data > 0, 1, self.slope.data[:,np.newaxis])
        else:
            raise("size input should be an integer >= 1")
