from mnist import MNIST
import numpy as np
import random
import json

def load_data():
    mndata = MNIST('./MNIST_ORG')
    images, labels = mndata.load_training()
    images_t,labels_t = mndata.load_testing()
    images = [np.reshape(i, (784, 1))/255 for i in images]
    labels = [vector_output(o) for o in labels]
    images_t = [np.reshape(i, (784, 1)) for i in images_t]
    training_data = list(zip(images, labels))
    test_data = list(zip(images_t, labels_t))
    return training_data, test_data

def vector_output(o):
    temp = np.zeros((10,1))
    temp[o] = 1.0
    return temp

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]

    def feedforword(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        random.shuffle(training_data)
        for i in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_network(mini_batch, eta)
            self.save("data.json")
            if test_data:
                print("Epoch ", i, ": ", self.evaluate(test_data), " / ", len(test_data))
            else:
                print("Epoch ", i, " complete")
    
    def update_network(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        derivative = (activations[-1]-y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = derivative
        nabla_w[-1] = np.dot(derivative, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            derivative = np.dot(self.weights[-l+1].transpose(), derivative) * sigmoid_prime(z)
            nabla_b[-l] = derivative
            nabla_w[-l] = np.dot(derivative, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforword(x)), y) for (x,y) in test_data]
        return sum(int(x==y) for x,y in test_results)
    
    def save(self, filename):
        data = {"sizes":self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

training_data, test_data = load_data()
net = Network([784,100,30,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)