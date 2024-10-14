# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from scipy import misc,ndimage
import matplotlib.pyplot as plt



mnist = input_data.read_data_sets('./MNIST_data')
# mnist=tf.keras.datasets.mnist.load_data('./MNIST_data')

batch_size = 100
width,height = 28,28
mnist_dim = width*height
random_dim = 10
epochs = 100000

def my_init(size):
    return tf.random_uniform(size, -0.05, 0.05)

D_W1 = tf.Variable(my_init([mnist_dim, 128]))
D_b1 = tf.Variable(tf.zeros([128]))
D_W2 = tf.Variable(my_init([128, 32]))
D_b2 = tf.Variable(tf.zeros([32]))
D_W3 = tf.Variable(my_init([32, 1]))
D_b3 = tf.Variable(tf.zeros([1]))
D_variables = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

G_W1 = tf.Variable(my_init([random_dim, 32]))
G_b1 = tf.Variable(tf.zeros([32]))
G_W2 = tf.Variable(my_init([32, 128]))
G_b2 = tf.Variable(tf.zeros([128]))
G_W3 = tf.Variable(my_init([128, mnist_dim]))
G_b3 = tf.Variable(tf.zeros([mnist_dim]))
G_variables = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

def D(X):
    X = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    X = tf.nn.relu(tf.matmul(X, D_W2) + D_b2)
    X = tf.matmul(X, D_W3) + D_b3
    return X

def G(X):
    X = tf.nn.relu(tf.matmul(X, G_W1) + G_b1)
    X = tf.nn.relu(tf.matmul(X, G_W2) + G_b2)
    X = tf.nn.sigmoid(tf.matmul(X, G_W3) + G_b3)
    return X

real_X = tf.placeholder(tf.float32, shape=[batch_size, mnist_dim])
random_X = tf.placeholder(tf.float32, shape=[batch_size, random_dim])
random_Y = G(random_X)

# 定义一个均匀分布的随机变量，用于在真实图片和生成图片之间插值
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
# 计算插值后的图片
X_inter = eps*real_X + (1. - eps)*random_Y
# 计算判别器在插值图片上的梯度
grad = tf.gradients(D(X_inter), [X_inter])[0]
# 计算梯度的范数
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
# 计算梯度惩罚项，它使得判别器在接近真实图片时梯度为1
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

#判别器希望判断生成的y的越大越好，判断真实样本越小越好
D_loss = tf.reduce_mean(D(real_X)) - tf.reduce_mean(D(random_Y)) + grad_pen
#生成器让判别器判断生成的y的越小越好
G_loss = tf.reduce_mean(D(random_Y))

D_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(D_loss, var_list=D_variables)
G_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(G_loss, var_list=G_variables)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')


for e in range(epochs):
    for i in range(5):
        real_batch_X,_ = mnist.train.next_batch(batch_size)
        random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim)) #生成输入样本X
        _,D_loss_ = sess.run([D_solver,D_loss], feed_dict={real_X:real_batch_X, random_X:random_batch_X}) #训练D
    random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))  #生成输入样本X
    _,G_loss_ = sess.run([G_solver,G_loss], feed_dict={random_X:random_batch_X}) #训练G
    if e % 10000 == 0:
        print ('epoch %s, D_loss: %s, G_loss: %s'%(e, D_loss_, G_loss_))
        n_rows = 6
        check_imgs = sess.run(random_Y, feed_dict={random_X:random_batch_X}).reshape((batch_size, width, height))[:n_rows*n_rows]
        imgs = np.ones((width*n_rows+5*n_rows+5, height*n_rows+5*n_rows+5))
        for i in range(n_rows*n_rows):
            imgs[5+5*(i%n_rows)+width*(i%n_rows):5+5*(i%n_rows)+width+width*(i%n_rows), 5+5*(i//n_rows)+height*(i//n_rows):5+5*(i//n_rows)+height+height*(i//n_rows)] = check_imgs[i]
        plt.imsave('out/%s.png' % (e // 10000), imgs, cmap='gray')  # 如果是灰度图像，可以指定 cmap

