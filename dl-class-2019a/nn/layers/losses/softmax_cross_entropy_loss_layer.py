import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """
        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """
<<<<<<< HEAD
        :param logits: ND non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets
        :param targets: (N-1)D class id integers.
=======

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
>>>>>>> 3d2ae07544fcf84919364d78ac33393f8333b4c1
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        N = logits.shape[0]
        
        logits -= np.max(logits, axis=axis, keepdims=True)
        log_softmax = logits - np.log(np.sum(np.exp(logits), axis=axis, keepdims=True))
        loss = -1 * np.sum(log_softmax[np.arange(N), targets[:]])

        if self.reduction == 'mean':
            loss /= N
        elif self.reduction == 'sum':
            pass

        self.prev_data = (log_softmax, targets)

        return loss

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        log_softmax, targets = self.prev_data
        N = targets.shape[0]
        dx = np.exp(log_softmax)
        dx[np.arange(N), targets] -= 1

        if self.reduction == 'mean':
            dx /= N
        elif self.reduction == 'sum':
            pass

        return dx
