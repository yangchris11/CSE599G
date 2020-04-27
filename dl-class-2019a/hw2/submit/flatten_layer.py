import numpy as np
from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)

        self.previous_data = None

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        self.previous_data = data
        x = data.reshape(data.shape[0], -1)

        return x

    def backward(self, previous_partial_gradient):
        # TODO
        return previous_partial_gradient.reshape(self.previous_data.shape)
