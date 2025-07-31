import numpy as np
from importlib import reload as imp_reload

def reload(lib):
    imp_reload(lib)


class Linear():
    def __init__(self, input_size, output_size, bias = True):
        self.input_size = input_size
        self.output_size = output_size
        self.is_bias = bias

        self.weight = np.zeros([self.output_size, self.input_size])
        if self.is_bias:
            self.bias = np.zeros([self.output_size, 1])
        self.initialize_layers()

        self.input = 0
        self.output = 0
        self.grad = 0
        self.dl_db = 0
        self.dl_dw = 0
        self.velocity_bias = 0
        self.velocity_weight = 0

    # kaiming init
    def initialize_layers(self):
        self.weight = np.random.normal(0, np.sqrt(2/ self.input_size), [self.output_size, self.input_size])
        self.weight = self.weight.astype(np.float32)

        if self.is_bias:
            self.bias = np.zeros([self.output_size, 1]) # in he initalization biases are just init to zero
            self.bias = self.bias.astype(np.float32)

    def load_weights(self, weight, bias = None):
        if self.is_bias and bias is not None:
            if bias.shape != self.bias.shape:
                raise ValueError(f"Bias shape {bias.shape} does not match layer shape {self.bias.shape}")
            self.bias = bias.astype(np.float32)
        if weight.shape != self.weight.shape:
            raise ValueError(f"Weight shape {weight.shape} does not match layer shape {self.weight.shape}")
        self.weight = weight.astype(np.float32)

    def __call__(self, input):
        self.input = input
        x = self.weight @ self.input
        if self.is_bias:
            x += self.bias
        return x
    
    def backprop(self, prop_grad):
        self.grad = prop_grad
        return self.grad @ self.weight 

    
    def update(self, gamma, learning_rate):
        if self.is_bias:
            self.dl_db = np.mean(self.grad, axis=0).reshape(-1, 1)
            self.velocity_bias = gamma * self.velocity_bias + (1 - gamma) * self.dl_db
            self.bias -= learning_rate * self.velocity_bias
        self.dl_dw = np.mean(np.einsum('bi,bj->bij', self.grad, self.input.T), axis=0)
        self.velocity_weight = gamma * self.velocity_weight + (1 - gamma) * self.dl_dw
        self.weight -= learning_rate * self.velocity_weight

    def __str__(self):
        return f"Linear(in_features={self.input_size}, out_features={self.output_size}, bias={self.is_bias})"
    
class ReLU():
    def __init__(self):
        self.input = 0
        self.output = 0
        self.grad = 0

    def __call__(self, input):
        x = np.where(input < 0, 0, input)
        self.input = input
        return x
    
    def backprop(self, prop_grad):
        self.grad = np.where(self.input < 0, 0, 1).T
        return np.multiply(prop_grad, self.grad)
    
    def __str__(self):
        return "ReLU()"

class Softmax():
    def __init__(self):
        self.input = 0
        self.output = 0
        self.grad = 0

    def __call__(self, input):
        x = input.T
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.output = x
        return x.T

    def backprop(self, prop_grad):
        dL_dYhat = prop_grad
        dot = np.sum(self.output * dL_dYhat, axis=1, keepdims=True)
        self.grad = self.output * (dL_dYhat - dot) # Note that this doesn't store the local gradient like the other layers but the propagated gradient
        return self.grad
    
    def __str__(self):
        return "Softmax()"


class Module():
    def __init__(self):
        self.cache = []

    def __call__(self, input):
        x = input.T
        for l in self.cache:
            x = l(x)
        return x.T


class CrossEntropyLoss():
    def __init__(self):
        self.y_pred = 0
        self.y_hat = 0
        self.grad = 0
        

    def __call__(self, y_pred, labels):
        C = y_pred.shape[1]
        N = y_pred.shape[0]
        self.y_hat = np.eye(C)[labels]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        log_pred = np.log(self.y_pred)
        return -1 * np.sum(np.multiply(self.y_hat, log_pred)) / N

    def backward(self):
        self.grad = - self.y_hat / self.y_pred
        return self.grad

class SGD():
    def __init__(self, model, lr, momentum = 0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
    
    def zero_grad(self):
        for l in self.model.cache:
            l.grad = 0

    def step(self, loss):
        for idx, l in enumerate(self.model.cache[::-1]):
            # print(idx, l, loss)
            if type(l) == Softmax:
                prop_grad = loss.grad
            prop_grad = l.backprop(prop_grad)
            if type(l) == Linear:
                l.update(self.momentum, self.lr)
        
    
class DataLoader():
    def __init__(self, data, labels, batch_size, shuffle):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0
        self.batches = self.create_batches()

    def create_batches(self):
        if self.shuffle:
            idx = np.random.permutation(len(self.data))
        else:
            idx = np.arange(len(self.data))
        batches = []
        for i in range(0, len(self.data), self.batch_size):
            batches.append((self.data[idx[i:i+self.batch_size]], self.labels[idx[i:i+self.batch_size]]))
        return batches
        

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < len(self.batches):
            self.idx += 1
            return self.batches[self.idx-1]
        else:
            if self.shuffle:
                self.batches = self.create_batches()
            self.idx = 0
            raise StopIteration
        
    def __len__(self):
        return len(self.data)