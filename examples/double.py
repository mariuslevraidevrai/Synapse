import synapse
import numpy

x = numpy.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
y = numpy.array([[0],[2],[4],[6],[8],[10],[12],[14],[16],[18]])

net = synapse.NeuralNetwork(layers=[1,3,1],activation="linear")
net.train(x,y,iterations=100000)

result = net.forward([10])
print(result)