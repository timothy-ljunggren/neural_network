from mnist import MNIST
import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def load_data():
    mndata = MNIST('./MNIST_ORG')
    images, labels = mndata.load_training()
    images_t,labels_t = mndata.load_testing()
    images = [np.reshape(i, (784, 1)) for i in images]
    labels = [vector_output(o) for o in labels]
    images_t = [np.reshape(i, (784, 1)) for i in images_t]
    labels_t = [vector_output(o) for o in labels_t]
    return images, labels, images_t, labels_t

def vector_output(o):
    temp = np.zeros((10,1))
    temp[o] = 1.0
    return temp

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1])]
    
    def backprop(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def train(self, training_data, labels, iterations, batch_size, eta):
        shuffle = np.arange(len(training_data))
        random.shuffle(shuffle)
        for i in range(iterations):
            mini_batches = [(training_data[x],labels[x]) for x in shuffle[i:i+batch_size]]
            for mini_batch in mini_batches:
                self.update_network(mini_batch, eta)
            print("Epoch ",i," complete")
    
    def update_network(self, mini_batch, eta):
        pass




images, labels, images_t, labels_t = load_data()
net = Network([784,30,10])
print(net.backprop(images_t[0]))
print(labels_t[0])