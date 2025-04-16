import numpy as np
from activations import Linear

class Layer:
    def __init__(self, learning_rate, activation, name):
        self.learning_rate = learning_rate
        self.activation = activation
        self.name = name
        self.grads = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self):
        pass

    def reset_cache(self):
        self.grads.clear()

class Input(Layer):
    def __init__(self, input_size):
        super().__init__(0, Linear(), "Input")

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

class Dense(Layer):
    def __init__(self, input_size, output_size, activation, learning_rate):
        super().__init__(learning_rate, activation, 'Dense')
        self.weights = xavier_init(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.inputs = None
        self.outputs = []

    def forward(self, x):
        self.inputs = x
        z = x @ self.weights + self.bias
        out = self.activation.forward(z)
        self.outputs.append(out)
        return out

    def backward(self, grad):
        output = self.outputs.pop()
        delta = self.activation.backward(output) * grad
        self.grads["weights"] = self.inputs.T @ delta
        self.grads["bias"] = np.sum(delta, axis=0)
        return delta @ self.weights.T

    def update(self):
        self.weights -= self.learning_rate * self.grads["weights"]
        self.bias -= self.learning_rate * self.grads["bias"]

class RNN(Layer):
    def __init__(self, input_size, hidden_size, learning_rate, activation):
        super().__init__(learning_rate, activation, 'RNN')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = xavier_init(input_size, hidden_size)
        self.W_hh = xavier_init(hidden_size, hidden_size)
        self.bias = np.zeros(hidden_size)
        self.h_prev = None
        self.inputs, self.hiddens = [], []

    def forward(self, x):
        h_prev = self.hiddens[-1] if self.hiddens else np.zeros((x.shape[0], self.hidden_size))
        self.inputs.append(x)
        self.h_prev = h_prev
        h = self.activation.forward(x @ self.W_ih + h_prev @ self.W_hh + self.bias)
        self.hiddens.append(h)
        return h

    def backward(self, grad):
        h = self.hiddens.pop()
        x = self.inputs.pop()
        dh = self.activation.backward(h) * grad

        self.grads['W_ih'] = x.T @ dh
        self.grads['W_hh'] = self.h_prev.T @ dh
        self.grads['bias'] = np.sum(dh, axis=0)

        dx = dh @ self.W_ih.T
        self.h_prev = None
        return dx

    def update(self):
        self.W_ih -= self.learning_rate * self.grads['W_ih']
        self.W_hh -= self.learning_rate * self.grads['W_hh']
        self.bias -= self.learning_rate * self.grads['bias']

def xavier_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
