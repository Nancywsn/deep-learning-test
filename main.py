import mnist_reader
import numpy as np
import pandas as pd
from model import NeuralNetwork
from optimizer import SGD
import visualization

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

def trans_y(y):
    # 获取 unique 类别数量
    num_classes = len(np.unique(y))
    # 创建 one-hot 编码矩阵
    y_one_hot = np.zeros((y.shape[0], num_classes))
    # 将 y_train 转换为 one-hot 编码
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot

def normaliz(X):
    # 计算每列的最大值和最小值
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    # 进行标准化
    normalized_dataset = (X - min_vals) / (max_vals - min_vals)
    return normalized_dataset

# 自定义参数
layers = [[784, 32, 10], [784, 64, 10], [784, 128, 10]]
weight_decaies = [0, 1e-2, 2e-2] # L2 Penalty
learning_rates = [5e-3, 1e-2, 2e-2]
batch_size = 16
epochs= 30

# 训练模型
print('Training models...')
best_config = {'accuracy': 0}
for layer in layers:
    for learning_rate in learning_rates:
        for weight_decay in weight_decaies:
            print(f"**Current layer: {layer}, Current learning rate: {learning_rate}, Current weight decay: {weight_decay}")
            nn = NeuralNetwork(layer)
            optimizer = SGD(nn, learning_rate, weight_decay, batch_size)
            accuracy = nn.train(normaliz(X_train), trans_y(y_train), normaliz(X_test), trans_y(y_test), epochs, optimizer)
            if accuracy > best_config['accuracy']:
                best_config['accuracy'] = accuracy
                best_config['layer'] = layer
                best_config['learning_rate'] = learning_rate
                best_config['weight_decay'] = weight_decay
             
print(best_config)
# {'accuracy': 88.64999999999999, 'layer': [784, 128, 10], 'learning_rate': 0.02, 'weight_decay': 0}

# 加载模型
print("Testing...")
nn = NeuralNetwork(best_config['layer'])
nn.load(f"model_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.npz")
nn.test(normaliz(X_test), trans_y(y_test))

# 可视化
print("Visualizing...")
log = pd.read_csv(f"logs/log_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.csv")
visualization.visualize(nn, log)