# Copyright (C) 2026  Synapse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from tqdm import tqdm


def showVersion():
    print("""⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣤⡤⣄⣤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣠⣴⣾⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⣾⣿⣦⣤⡀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣴⣾⢧⣽⣻⠿⢿⠿⠻⣯⣿⠾⢿⡿⣿⣼⡾⠾⣿⣯⣿⣶⡄⠀⠀⠀⠀
⠀⢠⢾⣷⡽⣻⣻⣿⣿⣿⣿⣿⣿⣏⣶⣯⢷⣿⣿⣏⢾⣯⣿⣿⣿⣿⣦⠀⠀⠀
⢰⣿⣿⣿⣿⢻⣿⢿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣟⣧⣼⣿⣿⣿⢕⣽⣾⣧⠀⠀
⠺⣽⣿⣿⣏⣾⣾⣿⣿⣿⣿⣿⣿⣾⣿⣿⣾⡿⣻⣿⣿⣿⣿⣾⣿⣿⣿⣽⡇⠀
⠀⠻⣽⣿⣾⣻⣿⣿⣿⣿⣿⣿⣿⣟⣿⣿⣹⣾⣷⣿⣯⣿⣾⣿⣿⣻⣿⣿⣿⣦
⠀⠀⠈⠛⠿⠿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⡯⣻⣿⣿⣿⣿⢽⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⢸⡸⢿⣳⣽⣿⣿⣿⣿⣼⣿⣽⣷⣝⣝⣿⣾⣿⣿⣿⣿⣿⣭
⠀⠀⠀⠀⠀⠀⠀⠈⠛⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠏⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠛⠙⢯⣿⣿⣿⢿⣿⣿⡷⣿⣴⡮⣿⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢹⣿⣿⣿⣿⣿⣿⢿⣟⡿⠋⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⢿⠙⠾⠿⠿⠟⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀""")
    print("This is Synapse v0.0.1, built for x86_64")


class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', lr=0.001, normalize=True, task='regression'):
        self.task = task
        self.useNormalization = normalize
        self.layers = layers
        self.lr = lr
        self.xMin = None
        self.xMax = None
        self.yMin = None
        self.yMax = None
        self.activationName = activation
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            scaler = np.sqrt(2.0 / layers[i]) if activation == 'relu' else np.sqrt(1.0 / layers[i])
            w = np.random.randn(layers[i], layers[i + 1]) * scaler
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activationDerivative = self.derivedSigmoid
        elif activation == 'relu':
            self.activation = self.ReLU
            self.activationDerivative = self.derivedReLU
        else:
            self.activation = lambda x: x
            self.activationDerivative = lambda x: np.ones_like(x)

    def save(self, path):
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'layers': self.layers,
            'lr': self.lr,
            'task': self.task,
            'activationName': self.activationName,
            'useNormalization': self.useNormalization,
            'xMin': self.xMin,
            'xMax': self.xMax,
            'yMin': self.yMin,
            'yMax': self.yMax,
        }
        np.save(path, data)
    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True).item()
        
        net = cls(
            layers=data['layers'],
            activation=data['activationName'],
            lr=data['lr'],
            normalize=data['useNormalization'],
            task=data['task']
        )
        net.weights = data['weights']
        net.biases = data['biases']
        net.xMin = data['xMin']
        net.xMax = data['xMax']
        net.yMin = data['yMin']
        net.yMax = data['yMax']
        
        return net
    def sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def derivedSigmoid(self, x):
        return x * (1 - x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def derivedReLU(self, x):
        return (x > 0).astype(float)

    def normalize(self, x, xMin, xMax):
        xMin = np.array(xMin, dtype=np.float64)
        xMax = np.array(xMax, dtype=np.float64)
        return (x - xMin) / (xMax - xMin + 1e-8)

    def denormalize(self, xNorm, xMin, xMax):
        xMin = np.array(xMin, dtype=np.float64)
        xMax = np.array(xMax, dtype=np.float64)
        return xNorm * (xMax - xMin) + xMin
    
    def forwardInternal(self, x):
        self.activations = [x]
        self.zValues = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w) + b
            self.zValues.append(z)
            isLast = (i == len(self.weights) - 1)
            if isLast:
                if self.task == 'binaryClassification':
                    a = self.sigmoid(z)
                else:
                    a = z
            else:
                a = self.activation(z)
            self.activations.append(a)
        return a

    def forward(self, x):
        xArray = np.array(x, dtype=np.float64)
        if xArray.ndim == 1:
            xArray = xArray.reshape(1, -1)

        if self.useNormalization and self.xMin is not None:
            xNorm = self.normalize(xArray, self.xMin, self.xMax)
        else:
            xNorm = xArray

        yNorm = self.forwardInternal(xNorm)

        if self.task == 'binaryClassification':
            return yNorm

        if self.useNormalization and self.yMin is not None:
            y = self.denormalize(yNorm, self.yMin, self.yMax)
        else:
            y = yNorm
        return y
    
    def backward(self, y, output):
        m = y.shape[0]

        delta = (output - y) / m
        for i in reversed(range(len(self.weights))):
            aPrev = self.activations[i]
            dw = np.dot(aPrev.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            self.weights[i] -= self.lr * dw
            self.biases[i]  -= self.lr * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.activationDerivative(self.activations[i])
                delta = np.clip(delta, -1.0, 1.0)
                
    def train(self, x, y, iterations=5000):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.xMin = np.min(x, axis=0)
        self.xMax = np.max(x, axis=0)
        self.yMin = np.min(y)
        self.yMax = np.max(y)

        if self.useNormalization:
            xNorm = self.normalize(x, self.xMin, self.xMax)
        else:
            xNorm = x

        if self.task == 'binaryClassification':
            yTrain = y
        else:
            yTrain = self.normalize(y, self.yMin, self.yMax) if self.useNormalization else y

        pbar = tqdm(range(iterations), desc="Training")
        for i in pbar:
            output = self.forwardInternal(xNorm)
            self.backward(xNorm, yTrain, output)

            if i % 500 == 0 or i == iterations - 1:
                if self.task == 'binaryClassification':
                    loss = -np.mean(
                        yTrain * np.log(output + 1e-8) +
                        (1 - yTrain) * np.log(1 - output + 1e-8)
                    )
                else:
                    loss = np.mean((yTrain - output) ** 2)
                pbar.set_description(f"Iteration {i:>6}  Loss {loss:.6f}")