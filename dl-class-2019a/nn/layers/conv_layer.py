from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding_size = (kernel_size - 1) // 2
        self.stride = stride
        
        self.initialize()

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
    def forward_numba(data, weights, bias, N, C, H, W, k, s):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        x = np.zeros((N, C, H, W))

        for c in prange(C):
            cout_w = weights[:, c, :, :]
            # print('(forward_numba) w shape:', cout_w.shape)
            for n in prange(N):
                for h in prange(H):
                    for w in prange(W):
                        prev_x = data[n, :, h*s:h*s+k, w*s:w*s+k]
                        # print('(forward_numba) x shape:', prev_x.shape)
                        x[n, c, h, w] += (prev_x * cout_w).sum()
                x[n, c] += bias[c]

        assert x.shape == (N, C, H, W)
                        
        return x

    def forward(self, data):
        # TODO
        N, C, H, W = data.shape
        print('(forward) input data shape: {} x {} x {} x {}'.format(N, C, H, W))
        C_in, C_out, _, _ = self.weight.data.shape
        H_out = self.get_output_kernel_size(H)
        W_out = self.get_output_kernel_size(W)
        print('(forward) output data shape: {} x {} x {} x {}'.format(N, C_out, H_out, W_out))
        
        self.previous_data = data
        self.padded_data = self.get_padded_data(data)

        print('(forward) padded data shape:', self.padded_data.shape)
        print('(forward) weight shape:', self.weight.data.shape)
        print('(forward) bias shape:', self.bias.data.shape)

        x = self.forward_numba(
            self.padded_data, self.weight.data, self.bias.data,
            N, C_out, H_out, W_out, self.kernel_size, self.stride
        )

        return x

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad, N, C, H, W, k, s, p):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        x_grad = np.zeros(data.shape)
        print('(backward_numba) data gradient shape:', x_grad.shape)
        w_grad = kernel_grad
        print('(backward_numba) kernel gradient shape:', w_grad.shape)
        b_grad = np.zeros((N, C))
        print('(backward_numba) bias gradient shape:', b_grad.shape)

        # for n in range(N):
        #     for h in range(H):
        #         for w in range(W):
        #             for c in range(C):
        #                 grad = previous_grad[n ,c, h, w]
        #                 x_grad[n,:,h*s:h*s+k,w*s:w*s+k] += kerel[c] * grad

        # for n in range(N):
        #     for c in range(C):
        #         for h in range(H):
        #             for w in range(W):
        #                 grad = previous_grad[n, c, h, w]
        #                 w_grad[n,c] += data[n,:,h*s:h*s+k,w*s:w*s+k] * grad

        for n in prange(N):
            for c in prange(C):
                cout_w = kernel[c]
                for h in prange(H):
                    for w in prange(W):
                        grad = previous_grad[n ,c, h, w]

                        x_grad[n,:,h*s:h*s+k,w*s:w*s+k] += cout_w * grad
                        w_grad[n,c] += data[n,:,h*s:h*s+k,w*s:w*s+k] * grad
                        b_grad[n,c] += grad

        # for h in prange(H):
        #     for w in prange(W):
        #         for n in prange(N):
        #             prev_x = data[n,:,h*s:h*s+k,w*s:w*s+k]
        #             for c in prange(C):
        #                 grad = previous_grad[n, c, h, w]
        #                 cout_w = kernel[c]

        #                 x_grad[n,:,h*s:h*s+k,w*s:w*s+k] += cout_w * grad
        #                 w_grad[n,c] += prev_x * grad
        #                 b_grad[n,c] += grad

        # H_in = data.shape[2]
        # W_in = data.shape[3]
        # unpadded_x_grad = x_grad[:, :, p + H_in, p + W_in]
        # group_w_grad = np.swapaxes(np.sum(w_grad, axis=0), 0, 1)

        return x_grad, w_grad, b_grad

    def backward(self, previous_partial_gradient):
        # TODO
        N, C, H, W = self.previous_data.shape
        print('(backward) previous data shape: {} x {} x {} x {}'.format(N, C, H, W))
        C_in, C_out, _, _ = self.weight.data.shape
        H_out = self.get_output_kernel_size(H)
        W_out = self.get_output_kernel_size(W)
        print('(backward) output gradient shape: {} x {} x {} x {}'.format(N, C_out, H_out, W_out))

        self.padded_data = self.get_padded_data(self.previous_data)
        print('(backward) padded data shape:', self.padded_data.shape)
        print('(backward) weight gradient shape:', self.weight.grad.shape)
        print('(backward) bias gradient shape:', self.bias.grad.shape)
        
        channel_first_weight = np.swapaxes(self.weight.data, 0, 1)
        print('(backward) channel-first-weight gradient shape:', channel_first_weight.shape)

        dx, dw, db = self.backward_numba(
            previous_partial_gradient, self.padded_data, 
            channel_first_weight, np.zeros((N,) + channel_first_weight.shape),
            N, C_out, H_out, W_out, self.kernel_size, self.stride, self.padding_size
        )
        
        # unpad the dx term    
        dx = dx[:,:,self.padding_size:self.padding_size+H,self.padding_size:self.padding_size+W]

        # sum the dw term along axis=0
        dw = np.sum(dw, axis=0)
        dw = np.swapaxes(dw, 0, 1)

        # sum the db term along axis=0
        db = np.sum(db, axis=0)

        # update the gradients
        self.weight.grad = dw
        self.bias.grad = db
        
        return dx


    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2], self.weight.data.shape[3],
            self.weight.data.shape[0], self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
