import numpy as np
import matplotlib.pyplot as plt
import os
from datagen import batch_iterator

class NeuralNetwork:
    def __init__(self, layers, loss, learning_rate):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        for layer in self.layers:
            layer.update()

    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    def fit(self, inputs, targets, validation_inputs, validation_targets, epochs, batch_size, verbose=False):
        if not os.path.exists('plots'):
            os.makedirs('plots')

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            batch_losses = []
            for xb, yb in batch_iterator(batch_size, inputs, targets):
                batch_loss = 0
                for i in range(xb.shape[1]):
                    pred = self.forward(xb[:, i])
                    loss_grad = self.loss.grad(pred, yb[:, i])
                    batch_loss += self.loss.loss(pred, yb[:, i])
                    self.backward(loss_grad)
                    self.update()
                    self.reset_cache()

                    if verbose:
                        print("Pred:", pred.flatten())
                        print("Target:", yb[:, i].flatten())
                        print("Grad:", loss_grad.flatten())

                batch_losses.append(batch_loss / xb.shape[1])

            train_loss = np.mean(batch_losses)
            val_loss = self.evaluate_loss(validation_inputs, validation_targets)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.savefig('plots/training_validation_loss.png')

    def evaluate_loss(self, inputs, targets):
        losses = []
        for xb, yb in batch_iterator(1, inputs, targets):
            pred = self.forward(xb[:, 0])
            losses.append(self.loss.loss(pred, yb[:, 0]))
            self.reset_cache()
        return np.mean(losses)

    def evaluate_accuracy(self, inputs, targets, threshold=0.5):
        correct, total = 0, 0
        for xb, yb in batch_iterator(1, inputs, targets):
            pred = self.forward(xb[:, 0])[-1]
            pred_bin = (pred > threshold).astype(int)
            target_bin = np.array(yb[:, 0][-1], dtype=int)
            correct += np.sum(pred_bin == target_bin)
            total += len(target_bin)
            print()
            print(f"Pred: {pred_bin} || Target: {target_bin}")
            print('=======================================================================')
            print()
            self.reset_cache()
        print(f"Accuracy: {(correct / total) * 100:.2f}%")