import pandas as pd
import dataset
from model import NeuralNetwork
import visualization
from sklearn.metrics import confusion_matrix


# 加载数据
X_train, X_test, y_train, y_test = dataset.load_mnist_data()
# 归一化数据
X_train, X_test = dataset.normalize_data(X_train, X_test)
# One-hot 编码标签
y_train_onehot, y_test_onehot = dataset.one_hot_encode_labels(y_train, y_test)

best_config = {'layer': [784,256,128, 10], 'learning_rate': 0.01, 'weight_decay': 0.01}

# 加载模型
print("Testing...")
nn = NeuralNetwork(best_config['layer'])
nn.load(f"model_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.npy.npz")

incorrect_cases, y_true, y_pred = nn.test(X_test, y_test_onehot)

cm = confusion_matrix(y_true, y_pred)
print(cm)

# 可视化案例
# visualization.visualize_cases(incorrect_cases, title="Incorrect Cases", num_images=20)
# 混肴矩阵
# visualization.plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], title='Confusion Matrix')

# # 可视化
# print("Visualizing...")
# log = pd.read_csv(f"logs/log_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.csv")
# visualization.visualize(nn, log)