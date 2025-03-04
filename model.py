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
        sizes: layers = [[784, 32, 10], [784, 64, 10], [784, 128, 10]]
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        # 初始化权重，输入层没有权重和偏置,权重是从第一个隐藏层开始的
        self.weights = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(sizes[1:], sizes[:-1])] # 存储神经网络中每一层的权重矩阵
        self.biases = [np.array([0])] + [np.random.randn(x, 1) for x in sizes[1:]] 
        self.linear_transforms = [np.zeros(bias.shape) for bias in self.biases]
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
                self.activations[i] = softmax(self.linear_transforms[i]) # 输出层使用 Softmax 激活函数
            else:
                self.activations[i] = relu(self.linear_transforms[i]) # 隐藏层使用 ReLU 激活函数
                # self.activations[i] = tanh(self.linear_transforms[i]) # 隐藏层使用 ReLU 激活函数
                # self.activations[i] = leaky_relu(self.linear_transforms[i]) # 隐藏层使用 ReLU 激活函数

        return self.activations[-1] # 返回输出层的激活值
    
    def backward(self, loss_gradient):
        '''
        反向传播   

        `loss_gradient`为损失函数的求导结果
        '''
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        # 输出层的梯度计算
        nabla_b[-1] = loss_gradient
        nabla_w[-1] = loss_gradient.dot(self.activations[-2].transpose())

        # 反向传播到隐藏层
        for layer in range(self.num_layers-2, 0, -1):
            loss_gradient = np.multiply(
                self.weights[layer+1].transpose().dot(loss_gradient),
                # leaky_relu_gradient(self.linear_transforms[layer])
                relu_gradient(self.linear_transforms[layer])
                # tanh_gradient(self.linear_transforms[layer])
            )
            nabla_b[layer] = loss_gradient
            nabla_w[layer] = loss_gradient.dot(self.activations[layer-1].transpose())
        
        return nabla_b, nabla_w
    
    def train(self, X, Y, Xtest, Ytest, epochs, optimizer):
        '''
        X: 训练集输入数据，形状为 (m, 784)，其中 m 是样本数量，784 是输入特征数量（例如，MNIST 数据集的图像展平后的像素数量）。
        Y: 训练集标签，形状为 (m, 10)，One-hot 编码后的标签。
        Xtest: 测试集输入数据，形状为 (n, 784)。
        Ytest: 测试集标签，形状为 (n, 10)。
        epochs: 训练的总轮数。
        optimizer: 优化器对象，用于更新网络权重和偏置。
        '''
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
            res = [] # 用于存储预测结果
            for i in range(n):
                input, label = Xtest[i], Ytest[i]
                input = input.reshape(784,1)
                label = label.reshape(10,1)
                output = self.forward(input)
                validate_loss += np.where(label==1, -np.log(output), 0).sum()
                res.append(np.argmax(output) == np.argmax(label)) 
            validate_loss /= n # 计算平均交叉熵损失
            validate_losses.append(validate_loss)
            accuracy = sum(res) / n * 100
            accuracies.append(accuracy)
            print(f"****Epoch {epoch+1}, accuracy {accuracy} %.")
            
            # 训练模型
            # 打乱数据集
            indices = np.arange(m) # 获取训练集索引
            np.random.shuffle(indices)
            X_shuffled = X[indices, :]
            Y_shuffled = Y[indices, :]
            train_loss = 0 # 交叉熵损失
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size, :] # 获取当前批次的输入
                Y_batch = Y_shuffled[i:i+batch_size, :] # 获取当前批次的标签
                optimizer.zero_grad()  # 清零优化器的梯度
                for j in range(X_batch.shape[0]):
                    input, label = X_batch[j], Y_batch[j]
                    input = input.reshape(784,1)
                    label = label.reshape(10,1)
                    output = self.forward(input)
                    train_loss  += np.where(label==1, -np.log(output), 0).sum()
                    loss_gradient = output - label  # 计算损失函数的梯度
                    delta_nabla_b, delta_nabla_w = self.backward(loss_gradient) # 反向传播
                    optimizer.update(delta_nabla_b, delta_nabla_w) # 更新优化器的梯度
                optimizer.step() # 更新网络权重和偏置
            train_loss /= len(X) # 计算平均训练集损失
            # print('train_loss：', train_loss)
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
    
    # def save(self, filename):
    #     np.savez_compressed(
    #         file=os.path.join(os.curdir, 'parameter', filename),
    #         weights0=self.weights[0],
    #         weights1=self.weights[1],
    #         weights2=self.weights[2],
    #         biases0=self.biases[0],
    #         biases1=self.biases[1],
    #         biases2=self.biases[2],
    #         linear_transforms0=self.linear_transforms[0],
    #         linear_transforms1=self.linear_transforms[1],
    #         linear_transforms2=self.linear_transforms[2],
    #         activations0=self.activations[0],
    #         activations1=self.activations[1],
    #         activations2=self.activations[2]
    #     )

    def save(self, filename):
        save_path = os.path.join(os.curdir, 'parameter', filename)
        save_dict = {}

        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'weights{i}'] = weight
            save_dict[f'biases{i}'] = bias
        
        # 保存线性变换和激活函数
        if hasattr(self, 'linear_transforms'):
            for i, linear_transform in enumerate(self.linear_transforms):
                save_dict[f'linear_transforms{i}'] = linear_transform
        
        if hasattr(self, 'activations'):
            for i, activation in enumerate(self.activations):
                save_dict[f'activations{i}'] = activation
        
        # 保存到文件
        np.savez_compressed(file=save_path, **save_dict)


        
    # def load(self, filename):
    #     npz_members = np.load(os.path.join(os.curdir, 'parameter', filename), allow_pickle=True)

    #     # print('test', npz_members['weights0'].shape, npz_members['weights1'].shape)
    #     self.weights = [npz_members['weights0']]+[npz_members['weights1']]+[npz_members['weights2']]
    #     self.biases = [npz_members['biases0']]+[npz_members['biases1']]+[npz_members['biases2']]

    #     self.sizes = [b.shape[0] for b in self.biases]
    #     self.num_layers = len(self.sizes)

    #     self.linear_transforms = [npz_members['linear_transforms0']]+[npz_members['linear_transforms1']]+[npz_members['linear_transforms2']]
    #     self.activations = [npz_members['activations0']]+[npz_members['activations1']]+[npz_members['activations2']]

    def load(self, filename):
        load_path = os.path.join(os.curdir, 'parameter', filename)
        npz_members = np.load(load_path, allow_pickle=True)
        
        # 动态加载权重和偏置
        self.weights = []
        self.biases = []
        for key in sorted(npz_members.keys()):
            if key.startswith('weights'):
                self.weights.append(npz_members[key])
            elif key.startswith('biases'):
                self.biases.append(npz_members[key])

        self.linear_transforms = []
        self.activations = []
        for key in sorted(npz_members.keys()):
            if key.startswith('linear_transforms'):
                self.linear_transforms.append(npz_members[key])
            elif key.startswith('activations'):
                self.activations.append(npz_members[key])
        
        # 更新网络结构信息
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

    def test(self, Xtest, Ytest):
        n = Xtest.shape[0]
        res = []
        # correct_cases = []
        incorrect_cases = []
        y_true = []
        y_pred = []

        for i in range(n):
            input, label = Xtest[i], Ytest[i]
            input = input.reshape(784, 1)
            label = label.reshape(10, 1)
            output = self.forward(input)
            prediction = np.argmax(output)
            true_label = np.argmax(label)
            y_true.append(true_label)
            y_pred.append(prediction)

            if prediction != true_label:
                incorrect_cases.append((input, true_label, prediction))

            res.append(prediction == true_label)

        accuracy = sum(res) / n * 100
        print(f"****Test accuracy {accuracy:.2f} %.")  # 打印测试准确率

        return incorrect_cases, y_true, y_pred

########################################################################################

def sigmoid(input):
    return 1/(1+np.exp(-input))


def sigmoid_gradient(input):
    return sigmoid(input) * (1-sigmoid(input))


def relu(input):
    return np.maximum(0, input)


def relu_gradient(input):
    return input > 0

def leaky_relu(input, alpha=0.01):
    return np.where(input >= 0, input, alpha * input)

def leaky_relu_gradient(input, alpha=0.01):
    return np.where(input >= 0, 1, alpha)

def tanh(input):
    return np.tanh(input)

def tanh_gradient(input):
    return 1 - np.tanh(input) ** 2

def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))