import synapse
import numpy

x = numpy.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
y = numpy.array([[0],[1],[4],[9],[16],[25],[36],[49],[64],[81]])

net = synapse.NeuralNetwork(layers=[1,3,1],activation="sigmoid",lr=0.1)
net.train(x,y,iterations=100000)

result = net.forward([10])
print(result)