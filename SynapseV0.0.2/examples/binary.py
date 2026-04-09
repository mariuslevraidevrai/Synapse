import numpy as np
from synapse import NeuralNetwork

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

net = NeuralNetwork(layers=[2, 4, 1], activation='relu', lr=0.1, task='binaryClassification')

net.train(x, y, iterations=10000)

print(f"Probabilité pour [1,0]: {net.forward([1,0])}")