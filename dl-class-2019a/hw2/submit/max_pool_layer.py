import numbers

import numpy as np

from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding_size = (kernel_size - 1) // 2
        self.stride = stride

        self.previous_data = None
        self.padded_data = None

    def get_output_kernel_size(self, x):
        return 1 + (x + 2 * self.padding_size - self.kernel_size) // self.stride

    def get_padded_data(self, x):
        return np.pad(
            x, ((0, 0), (0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size)),
            mode = 'constant', constant_values = 0
        )

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, N, C, H, W, k, s):
        # data is N x C x H x W
        # TODO
        x = np.zeros((N, C, H, W))

        for c in prange(C):
            for n in prange(N):
                for h in prange(H):
                    for w in prange(W):
                        pool_x = data[n, c, h*s:h*s+k, w*s:w*s+k]
                        x[n, c, h, w] = np.max(pool_x)
        
        return x

    def forward(self, data):
        # TODO
        N, C, H, W = data.shape
        print('(forward) input data shape: {} x {} x {} x {}'.format(N, C, H, W))
        H_out = self.get_output_kernel_size(H)
        W_out = self.get_output_kernel_size(W)
        print('(forward) output data shape: {} x {} x {} x {}'.format(N, C, H_out, W_out))
        
        self.previous_data = data
        self.padded_data = self.get_padded_data(data)

        x = self.forward_numba(
            self.padded_data, 
            N, C, H_out, W_out, self.kernel_size, self.stride
        )
        
        return x

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, N, C, H, W, k, s):
        # data is N x C x H x W
        # TODO
        x_grad = np.zeros(data.shape)
        print('(backward_numba) data gradient shape:', x_grad.shape)

        # for c in prange(C):
        #     for n in prange(N):
        #         for h in prange(H):
        #             for w in prange(W):
        #                 pool_x = data[n, c, h*s:h*s+k, w*s:w*s+k]
        #                 max_x = np.max(pool_x)
        #                 x_grad[n,c,h*s:h*s+k,w*s:w*s+k] += previous_grad[n,c,h,w] * (max_x == pool_x)

        for n in prange(N):
            for c in prange(C):
                for h in prange(H):
                    for w in prange(W):
                        pool_x = data[n, c, h*s:h*s+k, w*s:w*s+k]
                        max_x = np.max(pool_x)
                        x_grad[n,c,h*s:h*s+k,w*s:w*s+k] += previous_grad[n,c,h,w] * (max_x == pool_x)

        return x_grad

    def backward(self, previous_partial_gradient):
        # TODO
        N, C, H, W = self.previous_data.shape
        print('(backward) previous data shape: {} x {} x {} x {}'.format(N, C, H, W))
        H_out = self.get_output_kernel_size(H)
        W_out = self.get_output_kernel_size(W)
        print('(backward) output gradient shape: {} x {} x {} x {}'.format(N, C, H_out, W_out))

        self.padded_data = self.get_padded_data(self.previous_data)
        print('(backward) padded data shape:', self.padded_data.shape)

        dx = self.backward_numba(
            previous_partial_gradient, self.padded_data,
            N, C, H_out, W_out, self.kernel_size, self.stride
        )

        # unpad the dx term    
        dx = dx[:,:,self.padding_size:self.padding_size+H,self.padding_size:self.padding_size+W]

        return dx

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
