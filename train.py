import dataset
import numpy as np
import pandas as pd
from model import NeuralNetwork
from optimizer import SGD
import visualization

# 加载数据
X_train, X_test, y_train, y_test = dataset.load_mnist_data()
# 归一化数据
X_train, X_test = dataset.normalize_data(X_train, X_test)
# One-hot 编码标签
y_train_onehot, y_test_onehot = dataset.one_hot_encode_labels(y_train, y_test)


# 自定义参数
# layers = [[784, 32, 10], [784, 64, 10], [784, 128, 10]]
# weight_decaies = [0, 1e-2, 2e-2] # L2 Penalty
# learning_rates = [5e-3, 1e-2, 2e-2]
layers = [[784, 256, 128, 10]]
weight_decaies = [1e-4, 2e-2] # L2 Penalty
learning_rates = [1e-2, 0]
batch_size = 64
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
            accuracy = nn.train(X_train, y_train_onehot, X_test, y_test_onehot, epochs, optimizer)
            if accuracy > best_config['accuracy']:
                best_config['accuracy'] = accuracy
                best_config['layer'] = layer
                best_config['learning_rate'] = learning_rate
                best_config['weight_decay'] = weight_decay
             
print(best_config)

# best_config = {'accuracy': 88.64999999999999, 'layer': [784, 128, 10], 'learning_rate': 0.02, 'weight_decay': 0}

# 加载模型
# print("Testing...")
# nn = NeuralNetwork(best_config['layer'])
# nn.load(f"model_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.npy.npz")
# nn.test(X_test, y_test_onehot)

# 可视化
print("Visualizing...")
log = pd.read_csv(f"logs/log_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.csv")
visualization.visualize(nn, log)