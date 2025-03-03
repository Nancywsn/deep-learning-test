# 导入必要的库
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_mnist_data():
    """加载 MNIST 数据集并划分训练集和测试集"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.int8)  # 将标签转换为整数类型
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """对像素值进行归一化"""
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test

def one_hot_encode_labels(y_train, y_test):
    """对标签进行 One-hot 编码"""
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    return y_train_onehot, y_test_onehot

def visualize_data(X_train, y_train, num_images=10, save_path=None):
    """可视化部分训练数据"""
    fig, axes = plt.subplots(2, num_images // 2, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        img = X_train[i].reshape(28, 28)  # 将784维数据重新塑形为28x28的图像
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    plt.show()

def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_mnist_data()
    print("训练集数据形状:", X_train.shape)
    print("训练集标签形状:", y_train.shape)
    print("测试集数据形状:", X_test.shape)
    print("测试集标签形状:", y_test.shape)
    
    # 归一化数据
    X_train, X_test = normalize_data(X_train, X_test)
    print("训练集像素值范围:", X_train.min(), X_train.max())
    print("测试集像素值范围:", X_test.min(), X_test.max())
    
    # One-hot 编码标签
    y_train_onehot, y_test_onehot = one_hot_encode_labels(y_train, y_test)
    print("训练集 One-hot 编码后的标签形状:", y_train_onehot.shape)
    print("测试集 One-hot 编码后的标签形状:", y_test_onehot.shape)
    
    # 可视化部分数据
    visualize_data(X_train, y_train, num_images=10, save_path='figs/train_img.png')

if __name__ == "__main__":
    main()