import numpy as np

class Activation:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc
        self.inputs = None

    def forward(self, x):
        self.inputs = x
        return self.func(x)

    def backward(self, grad):
        return self.dfunc(self.inputs) * grad

class Linear(Activation):
    def __init__(self):
        super().__init__(lambda x: x, lambda x: np.ones_like(x))

class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_deriv(x):
            s = sigmoid(x)
            return s * (1 - s)
        super().__init__(sigmoid, sigmoid_deriv)

class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
