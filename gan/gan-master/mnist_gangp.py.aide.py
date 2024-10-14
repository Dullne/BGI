# 导入tensorflow模块，并指定使用兼容性版本v1，禁用v2行为
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 从tensorflow的教程示例中导入处理MNIST数据集的方法
from tensorflow.examples.tutorials.mnist import input_data
# 导入os模块用于操作文件和目录，numpy用于数学计算，scipy的misc和ndimage用于图像处理，matplotlib用于绘图
import os
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

# 读取MNIST数据集，存储在当前目录下的MNIST_data文件夹中
mnist = input_data.read_data_sets('./MNIST_data')
# 使用Keras方式加载MNIST数据集的代码被注释掉了，这里我们使用传统的tensorflow方法

# 定义训练的批处理大小，图像的宽度和高度（均为28像素），以及MNIST数据的维度（宽*高）
batch_size = 100
width, height = 28, 28
mnist_dim = width * height
# 定义生成器输入的随机数据维度
random_dim = 10
# 定义训练的总周期数
epochs = 1#00000

# 定义初始化权重的函数，使用均匀分布随机初始化，并且权重大小在-0.05到0.05之间
def my_init(size):
    return tf.random_uniform(size, -0.05, 0.05)

# 初始化判别器的权重和偏置，包括三个全连接层的权重和偏置
D_W1 = tf.Variable(my_init([mnist_dim, 128]))
D_b1 = tf.Variable(tf.zeros([128]))
D_W2 = tf.Variable(my_init([128, 32]))
D_b2 = tf.Variable(tf.zeros([32]))
D_W3 = tf.Variable(my_init([32, 1]))
D_b3 = tf.Variable(tf.zeros([1]))
# 将判别器的变量放入一个列表中
D_variables = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

# 初始化生成器的权重和偏置，同样包括三个全连接层的权重和偏置
G_W1 = tf.Variable(my_init([random_dim, 32]))
G_b1 = tf.Variable(tf.zeros([32]))
G_W2 = tf.Variable(my_init([32, 128]))
G_b2 = tf.Variable(tf.zeros([128]))
G_W3 = tf.Variable(my_init([128, mnist_dim]))
G_b3 = tf.Variable(tf.zeros([mnist_dim]))
# 将生成器的变量放入一个列表中
G_variables = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

# 定义判别器模型，使用三个全连接层，激活函数为ReLU，不使用激活函数在输出层
def D(X):
    X = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    X = tf.nn.relu(tf.matmul(X, D_W2) + D_b2)
    X = tf.matmul(X, D_W3) + D_b3
    return X

# 定义生成器模型，使用三个全连接层，激活函数为ReLU和sigmoid
def G(X):
    X = tf.nn.relu(tf.matmul(X, G_W1) + G_b1)
    X = tf.nn.relu(tf.matmul(X, G_W2) + G_b2)
    X = tf.nn.sigmoid(tf.matmul(X, G_W3) + G_b3)
    return X

# 定义真实数据的占位符，形状为[batch_size, mnist_dim]
real_X = tf.placeholder(tf.float32, shape=[batch_size, mnist_dim])
# 定义随机噪声数据的占位符，形状为[batch_size, random_dim]
random_X = tf.placeholder(tf.float32, shape=[batch_size, random_dim])
# 生成器根据随机噪声生成假图片
random_Y = G(random_X)

# 定义一个均匀分布的随机变量，用于在真实图片和生成图片之间插值
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
print('砂页岩',eps)
# 计算插值后的图片
X_inter = eps * real_X + (1. - eps) * random_Y
# 计算判别器在插值图片上的梯度
grad = tf.gradients(D(X_inter), [X_inter])[0]
# 计算梯度的范数
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
# 计算梯度惩罚项，它使得判别器在接近真实图片时梯度为1
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

# 计算判别器的损失函数，包括真实图片的损失，生成图片的损失，```
# 以及梯度惩罚项
D_loss = tf.reduce_mean(D(real_X)) - tf.reduce_mean(D(random_Y)) + grad_pen
# 计算生成器的损失函数，目的是让判别器将生成的图片判断为真实图片
G_loss = tf.reduce_mean(D(random_Y))  #生成器让判别器判断的越小越好

# 定义优化器，使用Adam算法优化判别器损失函数，指定学习率和一阶矩的指数衰减率
D_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(D_loss, var_list=D_variables)
# 定义优化器，使用Adam算法优化生成器损失函数
G_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(G_loss, var_list=G_variables)

# 创建一个tensorflow的会话
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists('out/'):
    os.makedirs('out/')

# 开始训练，遍历所有周期
for e in range(epochs):
    # 在每个周期内进行多次迭代，这里为5次
    for i in range(5):
        # 从MNIST数据集中获取一个真实图片的批次
        real_batch_X,_ = mnist.train.next_batch(batch_size)
        # 生成一个随机噪声批次作为生成器的输入
        random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
        # 运行优化器，并计算判别器的损失
        _,D_loss_ = sess.run([D_solver,D_loss], feed_dict={real_X:real_batch_X, random_X:random_batch_X})
    # 生成一个随机噪声批次作为生成器的输入
    random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
    # 运行优化器，并计算生成器的损失
    _,G_loss_ = sess.run([G_solver,G_loss], feed_dict={random_X:random_batch_X})
    # 每隔10000个周期打印损失函数的值
    if e % 10000 == 0:
        print ('epoch %s, D_loss: %s, G_loss: %s'%(e, D_loss_, G_loss_))
        # 获取生成器生成的图片批次
        n_rows = 6
        check_imgs = sess.run(random_Y, feed_dict={random_X:random_batch_X}).reshape((batch_size, width, height))[:n_rows*n_rows]
        # 创建一个空白的大图像用于保存生成的图片
        imgs = np.ones((width*n_rows+5*n_rows+5, height*n_rows+5*n_rows+5))
        # 将生成的图片填充到空白大图像中
        for i in range(n_rows*n_rows):
            imgs[5+5*(i%n_rows)+width*(i%n_rows):5+5*(i%n_rows)+width+width*(i%n_rows), 5+5*(i//n_rows)+height*(i//n_rows):5+5*(i//n_rows)+height+height*(i//n_rows)] = check_imgs[i]
        # 保存生成的图片到文件
        plt.imsave('out/%s.png' % (e // 10000), imgs, cmap='gray')  # 如果是灰度图像，可以指定 cmap