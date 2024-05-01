import numpy as np


class SGD:

    def __init__(self, model, learning_rate, weight_decay, batch_size):
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.nabla_b = [np.zeros(bias.shape) for bias in self.model.biases]
        self.nabla_w = [np.zeros(weight.shape) for weight in self.model.weights]

    def zero_grad(self):
        self.nabla_b = [np.zeros(bias.shape) for bias in self.model.biases]
        self.nabla_w = [np.zeros(weight.shape) for weight in self.model.weights]

    def update(self, delta_nabla_b, delta_nabla_w):
        self.nabla_b = [nb + dnb for nb, dnb in zip(self.nabla_b, delta_nabla_b)]
        self.nabla_w = [nw + dnw for nw, dnw in zip(self.nabla_w, delta_nabla_w)]

    def step(self):
        # 参数更新
        # 使用batch gradient descent，因此学习率要除以batch_size
        self.model.weights = [(1 - self.lr * self.weight_decay) * w - (self.lr / self.batch_size) * dw for w, dw in zip(self.model.weights, self.nabla_w)]
        self.model.biases = [(1 - self.lr * self.weight_decay) * b - (self.lr / self.batch_size) * db for b, db in zip(self.model.biases, self.nabla_b)]