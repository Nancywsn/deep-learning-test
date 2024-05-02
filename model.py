import numpy as np
import pandas as pd
import os
import random

# 设置随机种子，确保结果可复现
np.random.seed(0)

class NeuralNetwork:
    
    # 初始化网络参数
    def __init__(self, sizes):
        '''
        初始化神经网络
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        # 初始化权重，输入层没有权重和偏置
        # [array(1,), array(32,784), array(10, 32)]
        self.weights = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(sizes[1:], sizes[:-1])]
        # [array(1), array(32), array(10)]
        self.biases = [np.array([0])] + [np.random.randn(x, 1) for x in sizes[1:]] 
        # 存储线性变换的结果: [array(1,), array(32,1), array(10, 1)]
        self.linear_transforms = [np.zeros(bias.shape) for bias in self.biases]
        # 存储非线性变换的结果: [array(1,), array(32,1), array(10, 1)]
        self.activations = [np.zeros(bias.shape) for bias in self.biases]
        
    def forward(self, input):
        '''
        前向传播
        input_dim: (784,1)
        output_dim: (10,1)
        '''
        self.activations[0] = input
        for i in range(1, self.num_layers):
            # 线性变换
            self.linear_transforms[i] = self.weights[i].dot(self.activations[i-1]) + self.biases[i]
            # 非线性变换
            # 在最后一层使用softmax激活函数
            if i == self.num_layers-1:
                self.activations[i] = softmax(self.linear_transforms[i])
            else:
                self.activations[i] = relu(self.linear_transforms[i])
        return self.activations[-1]
    
    def backward(self, loss_gradient):
        '''
        反向传播   

        `loss_gradient`为损失函数的求导结果
        '''
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        nabla_b[-1] = loss_gradient
        nabla_w[-1] = loss_gradient.dot(self.activations[-2].transpose())

        for layer in range(self.num_layers-2, 0, -1):
            loss_gradient = np.multiply(
                self.weights[layer+1].transpose().dot(loss_gradient),
                relu_gradient(self.linear_transforms[layer])
            )
            nabla_b[layer] = loss_gradient
            nabla_w[layer] = loss_gradient.dot(self.activations[layer-1].transpose())
        
        return nabla_b, nabla_w
    
    def train(self, X, Y, Xtest, Ytest, epochs, optimizer):
        batch_size = optimizer.batch_size
        n = Xtest.shape[0]
        m = X.shape[0]
        best_accuracy = 0
        train_losses = []
        validate_losses = []
        accuracies = []
        for epoch in range(epochs):
            # 测试模型
            validate_loss = 0 # 交叉熵损失
            res = []
            for i in range(n):
                input, label = Xtest[i], Ytest[i]
                input = input.reshape(784,1)
                label = label.reshape(10,1)
                output = self.forward(input)
                validate_loss += np.where(label==1, -np.log(output), 0).sum()
                res.append(np.argmax(output) == np.argmax(label))
            validate_loss /= n
            validate_losses.append(validate_loss)
            accuracy = sum(res) / n * 100
            accuracies.append(accuracy)
            print(f"****Epoch {epoch+1}, accuracy {accuracy} %.")
            
            # 训练模型
            # 打乱数据集
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices, :]
            Y_shuffled = Y[indices, :]
            train_loss = 0 # 交叉熵损失
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size, :]
                Y_batch = Y_shuffled[i:i+batch_size, :]
                optimizer.zero_grad()
                for j in range(X_batch.shape[0]):
                    input, label = X_batch[j], Y_batch[j]
                    input = input.reshape(784,1)
                    label = label.reshape(10,1)
                    output = self.forward(input)
                    train_loss  += np.where(label==1, -np.log(output), 0).sum()
                    loss_gradient = output - label
                    delta_nabla_b, delta_nabla_w = self.backward(loss_gradient)
                    optimizer.update(delta_nabla_b, delta_nabla_w)
                optimizer.step()
            train_loss /= len(X)
            # print('test', train_loss)
            train_losses.append(train_loss)
            # save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save(f"model_{self.sizes[1]}_{optimizer.lr}_{optimizer.weight_decay}.npy")
        # save log
        data = {
            "train_loss": train_losses,
            "validate_loss": validate_losses,
            "validate_accuracy": accuracies
        }
        pd.DataFrame(data).to_csv(f'logs/log_{self.sizes[1]}_{optimizer.lr}_{optimizer.weight_decay}.csv',)
        return best_accuracy
    
    def save(self, filename):
        np.savez_compressed(
            file=os.path.join(os.curdir, 'parameter', filename),
            weights0=self.weights[0],
            weights1=self.weights[1],
            weights2=self.weights[2],
            biases0=self.biases[0],
            biases1=self.biases[1],
            biases2=self.biases[2],
            linear_transforms0=self.linear_transforms[0],
            linear_transforms1=self.linear_transforms[1],
            linear_transforms2=self.linear_transforms[2],
            activations0=self.activations[0],
            activations1=self.activations[1],
            activations2=self.activations[2]
        )
        
    def load(self, filename):
        npz_members = np.load(os.path.join(os.curdir, 'models', filename), allow_pickle=True)

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        self.linear_transforms = list(npz_members['linear_transforms'])
        self.activations = list(npz_members['activations'])

    def test(self, Xtest, Ytest):
        n = Xtest.shape[0]
        res = []
        for i in range(n):
            input, label = Xtest[i], Ytest[i]
            input = input.reshape(784,1)
            label = label.reshape(10,1)
            output = self.forward(input)
            res.append(np.argmax(output) == np.argmax(label))
        accuracy = sum(res) / n * 100
        print(f"****Test accuracy {accuracy} %.")

########################################################################################

def sigmoid(input):
    return 1/(1+np.exp(-input))


def sigmoid_gradient(input):
    return sigmoid(input) * (1-sigmoid(input))


def relu(input):
    return np.maximum(0, input)


def relu_gradient(input):
    return input > 0


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))