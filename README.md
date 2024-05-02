# 从零开始构建三层神经网络分类器，实现图像分类
数据集[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

存放在本项目的data/fashion文件夹中
## 运行方式
```
python main.py
```
## 代码解释
* mnist_reader.py: 读取数据
* model.py: 模型训练
* optimizer.py: SGD优化器
* visualization.py: 模型参数可视化
* main.py: 主程序，查找最优超参数

  
* logs文件夹存放每次训练的模型参数，为可视化做准备
* figs文件夹存放可视化结果
* parameter文件夹保存模型权重
