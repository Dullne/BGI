from __future__ import print_function  # 用于兼容Python 2和3的print函数
import numpy as np  # 导入NumPy库，用于数值计算

class RBM:
    """
   .Restricted Boltzmann Machine (RBM) 类
    RBM是一种用于无监督学习的神经网络模型
    """
  
    def __init__(self, num_visible, num_hidden):
        """
        初始化RBM模型
        :param num_visible: 可见单元的数量
        :param num_hidden: 隐藏单元的数量
        """
        self.num_hidden = num_hidden  # 隐藏单元数量
        self.num_visible = num_visible  # 可见单元数量
        self.debug_print = True  # 是否打印调试信息

        # 初始化权重矩阵，维度为(num_visible x num_hidden)，使用均匀分布
        # 分布范围为[-sqrt(6. / (num_hidden + num_visible)), sqrt(6. / (num_hidden + num_visible))]
        # 这里初始化权重均值为0，标准差为0.1
        # 参考文献: Understanding the difficulty of training deep feedforward neural networks by Xavier Glorot and Yoshua Bengio
        np_rng = np.random.RandomState(1234)  # 创建一个随机数生成器，种子为1234

        #均匀分布
        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))
        print("初始权重：",self.weights.shape)

        # 在权重矩阵的第一行和第一列插入偏置单元的权重，初始化为0
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)
        print("加偏置权重：",self.weights.shape)
    def train(self, data, max_epochs=1000, learning_rate=0.1):
        """
        训练RBM模型
        :param data: 训练数据，每行是一个训练样本，包含可见单元的状态
        :param max_epochs: 最大训练轮数，默认为1000
        :param learning_rate: 学习率，默认为0.1
        """
        num_examples = data.shape[0]  # 训练样本的数量
        print("初始数据:",data.shape)
        # 在数据的第一列插入偏置单元，值为1
        data = np.insert(data, 0, 1, axis=1)
        print("加偏执数据:",data.shape)
        for epoch in range(max_epochs):
            # 正向传播，计算隐藏单元的激活值和概率
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            print(pos_hidden_probs)
            pos_hidden_probs[:, 0] = 1  # 固定偏置单元为1
            print(pos_hidden_probs)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # 计算正相关联
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # 反向传播，重构可见单元并再次计算隐藏单元的激活值和概率
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # 固定偏置单元为1
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # 计算负相关联
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # 更新权重
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            # 计算误差
            error = np.sum((data - neg_visible_probs) ** 2)
            # if self.debug_print:
            #     print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
        假设RBM已经训练好，根据可见单元的状态运行网络，获取隐藏单元的样本
        :param data: 每行是一个样本，包含可见单元的状态
        :return: 隐藏单元的状态矩阵
        """
        num_examples = data.shape[0]

        # 创建一个矩阵，每行是隐藏单元（包括偏置单元）的样本
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # 在数据的第一列插入偏置单元，值为1
        data = np.insert(data, 0, 1, axis=1)

        # 计算隐藏单元的激活值
        hidden_activations = np.dot(data, self.weights)
        # 计算隐藏单元的激活概率```python
        # 计算隐藏单元的激活概率
        hidden_probs = self._logistic(hidden_activations)
        # 根据概率随机决定隐藏单元的状态
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # 固定偏置单元为1
        hidden_states[:,0] = 1
  
        # 忽略偏置单元
        hidden_states = hidden_states[:,1:]
        return hidden_states
    
    # TODO: 是否可以移除与`run_visible`方法的代码重复？
    def run_hidden(self, data):
        """
        假设RBM已经训练好，根据隐藏单元的状态运行网络，获取可见单元的样本
        :param data: 每行是一个样本，包含隐藏单元的状态
        :return: 可见单元的状态矩阵
        """

        num_examples = data.shape[0]

        # 创建一个矩阵，每行是可见单元（包括偏置单元）的样本
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # 在数据的第一列插入偏置单元，值为1
        data = np.insert(data, 0, 1, axis=1)

        # 计算可见单元的激活值
        visible_activations = np.dot(data, self.weights.T)
        # 计算可见单元的激活概率
        visible_probs = self._logistic(visible_activations)
        # 根据概率随机决定可见单元的状态
        visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # 固定偏置单元为1
        visible_states[:,0] = 1

        # 忽略偏置单元
        visible_states = visible_states[:,1:]
        return visible_states
    
    def daydream(self, num_samples):
        """
        随机初始化可见单元一次，然后开始交替进行Gibbs采样步骤
        （每个步骤包括更新所有隐藏单元，然后更新所有可见单元），
        在每个步骤中采样可见单元的状态。
        注意，我们只初始化网络一次，所以这些样本是相关的。

        :return: 样本矩阵，每行是在网络幻想时产生的可见单元样本
        """

        # 创建一个矩阵，每行是可见单元（包括偏置单元）的样本，初始化为全1
        samples = np.ones((num_samples, self.num_visible + 1))

        # 从均匀分布中获取第一个样本
        samples[0,1:] = np.random.rand(self.num_visible)

        # 开始交替Gibbs采样
        # 注意，我们保持隐藏单元为二进制状态，但将可见单元保留为实数概率
        # 参见Hinton的“A Practical Guide to Training Restricted Boltzmann Machines”第3节
        for i in range(1, num_samples):
            visible = samples[i-1,:]

            # 计算隐藏单元的激活值
            hidden_activations = np.dot(visible, self.weights)      
            # 计算隐藏单元的激活概率
            hidden_probs = self._logistic(hidden_activations)
            # 根据概率随机决定隐藏单元的状态
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # 固定偏置单元为1
            hidden_states[0] = 1

            # 重新计算可见单元的激活概率
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_states

        # 忽略偏置单元（第一列），因为它们总是设置为1
        return samples[:,1:]        
      
    def _logistic(self, x):
        """
        logistic函数，也称为sigmoid函数
        :param x: 输入值
        :return: logistic函数的输出值
        """
        return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
    # 主函数，用于测试RBM模型
    r = RBM(num_visible=6, num_hidden=2)  # 创建一个RBM实例，可见单元6个，隐藏单元2个
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    r.train(training_data, max_epochs=5000)  # 使用训练数据进行训练，最大轮数为5000