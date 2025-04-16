import numpy as np


class Loss:
    def loss(self, y_pred, y):
        raise NotImplementedError

    def grad(self, y_pred, y):
        raise NotImplementedError


class MSE(Loss):
    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def grad(self, y_pred, y):
        return 2 * (y_pred - y)


class BinaryCrossEntropy:
    def loss(self, prediction, target):
        eps = 1e-9  # To avoid log(0)
        prediction = np.clip(prediction, eps, 1 - eps)  # Safe sigmoid output
        return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

    def grad(self, prediction, target):
        eps = 1e-9
        prediction = np.clip(prediction, eps, 1 - eps)
        return (prediction - target) / (prediction * (1 - prediction) * prediction.shape[0])