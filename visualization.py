import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def visualize(model, log):
    # 可视化训练和测试的loss曲线
    log[['train_loss','validate_loss']].plot(title='train/validate loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("figs/loss.png")
    plt.close()

    # 可视化测试的accuracy曲线
    log[['validate_accuracy']].plot(title='validate accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("figs/accuracy.png")
    plt.close()

    # 可视化每层的网络参数
    # layer1_weights = model.weights[1].flatten().tolist()
    # plt.hist(layer1_weights, bins=100)
    # plt.title("layer1 weights")
    # plt.xlabel("value")
    # plt.ylabel("frequency")
    # plt.savefig("figs/layer1_weights.png")
    # plt.close()

    # layer2_weights = model.weights[2].flatten().tolist()
    # plt.hist(layer2_weights, bins=30)
    # plt.title("layer2 weights")
    # plt.xlabel("value")
    # plt.ylabel("frequency")
    # plt.savefig("figs/layer2_weights.png")
    # plt.close()

    # layer1_biases = model.biases[1].flatten().tolist()
    # plt.hist(layer1_biases, bins=10)
    # plt.title("layer1 biases")
    # plt.xlabel("value")
    # plt.ylabel("frequency")
    # plt.savefig("figs/layer1_biases.png")
    # plt.close()

    # layer2_biases = model.biases[2].flatten().tolist()
    # plt.hist(layer2_biases, bins=10)
    # plt.title("layer2 biases")
    # plt.xlabel("value")
    # plt.ylabel("frequency")
    # plt.savefig("figs/layer2_biases.png")
    # plt.close()


def visualizelog(log):
    # 可视化训练和测试的loss曲线
    log[['train_loss','validate_loss']].plot(title='train/validate loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("figs/loss.png")
    plt.close()

    # 可视化测试的accuracy曲线
    log[['validate_accuracy']].plot(title='validate accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("figs/accuracy.png")
    plt.close()

def visualizelogs(logs):
    for string, log in logs.items():
        plt.plot(log['train_loss'], label=string)
        plt.plot(log['validate_loss'], label='validate'+string)
    
    # plt.title('Train Loss for Different Learning Rates')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("figs/comparison.png")
    plt.close()

def visualize_cases(cases, title="Visualization", num_images=10):
    """可视化案例"""
    n = min(len(cases), num_images)  # 限制最多显示 num_images 个案例
    fig, axes = plt.subplots(2, n // 2, figsize=(15, 5))
    axes = axes.flatten()

    for i in range(n):
        if i >= len(axes):
            break
        if len(cases[i]) == 2:  # 正确案例 (input, label)
            name = "Correct"
            input, label = cases[i]
            img = input.reshape(28, 28)  # 将输入数据重塑为28x28的图像
            axes[i].set_title(f"Label: {label}")
        elif len(cases[i]) == 3:  # 错误案例 (input, label, prediction)
            name = "Incorrect"
            input, label, prediction = cases[i]
            img = input.reshape(28, 28)  # 将输入数据重塑为28x28的图像
            axes[i].set_title(f"Label: {label}, Pred: {prediction}")
        else:
            raise ValueError(f"Unexpected case structure: {cases[i]}")

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"figs/{name}_cases.png")

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    # 手动设置颜色条范围，以增强对比度
    vmin = np.min(cm)
    vmax = np.max(cm)
    
    plt.figure(figsize=(12, 9))  # 增大热力图尺寸
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 12},  # 增大注释字体大小
                linewidths=0.5,  # 添加网格线
                cbar_kws={'shrink': 0.8},  # 调整颜色条大小
                vmin=vmin, vmax=vmax)  # 手动设置颜色条范围

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')  # 旋转X轴标签以便更好地显示
    plt.yticks(rotation=0)  # 保持Y轴标签水平
    plt.tight_layout()  # 调整布局
    plt.savefig("figs/confusion_matrix1.png")
    plt.close()

if __name__ == '__main__':
    # 可视化
    print("Visualizing...")

    # 可视化训练和测试的loss曲线
    # best_config = {'layer': [784, 256, 128, 10], 'learning_rate': 0.01, 'weight_decay': 0.0001}
    # log = pd.read_csv(f"logs/log_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.csv")
    # visualizelog(log)

    # # 可视化不同学习率的训练和测试的loss曲线
    logs = {
        # 'L2=0': pd.read_csv("logs/log_256_0.01_0.csv"),
        'L2=0.0001': pd.read_csv("logs/log_256_0.01_0.0001.csv"),
        'L2=0.001': pd.read_csv("logs/log_256_0.01_0.001.csv"),    
        # 'L2=0.1': pd.read_csv("logs/log_256_0.01_0.1.csv")    
    }
    visualizelogs(logs)

    # cm = np.array([
    #     [967, 0, 0, 0, 0, 3, 5, 3, 2, 0],
    #     [0, 1120, 2, 1, 0, 1, 4, 1, 6, 0],
    #     [6, 2, 999, 5, 4, 0, 2, 7, 7, 0],
    #     [0, 0, 7, 985, 1, 3, 0, 5, 6, 3],
    #     [1, 0, 3, 0, 950, 1, 6, 2, 2, 17],
    #     [8, 1, 0, 9, 2, 851, 7, 2, 8, 4],
    #     [6, 3, 1, 0, 7, 7, 929, 1, 4, 0],
    #     [2, 9, 13, 4, 3, 1, 0, 988, 1, 7],
    #     [6, 0, 1, 7, 5, 4, 5, 3, 940, 3],
    #     [6, 8, 1, 9, 18, 2, 0, 7, 4, 954]
    # ])

    # plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], title='Confusion Matrix')
