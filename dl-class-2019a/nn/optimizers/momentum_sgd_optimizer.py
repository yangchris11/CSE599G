from .base_optimizer import BaseOptimizer


class MomentumSGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate, momentum=0.9, weight_decay=0):
        super(MomentumSGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.previous_deltas = [0] * len(parameters)

    def step(self):
        for idx, parameter in enumerate(self.parameters):
            # TODO update the parameters
            delta = self.momentum * self.previous_deltas[idx] + parameter.grad + parameter.data * self.weight_decay
            self.previous_deltas[idx] = delta
            parameter.data -= self.learning_rate * delta
