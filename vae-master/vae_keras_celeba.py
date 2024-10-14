#! -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback

# 获取图像路径并打乱顺序
imgs = glob.glob('img_align_celeba/*.jpg')
np.random.shuffle(imgs)

# 获取图像的高度和宽度
with Image.open(imgs[0]) as img:
    width, height = img.size
    center_height = int((height - width) / 2)

img_dim = 64
z_dim = 512

# 自定义的 imread 函数
def imread(f):
    with Image.open(f) as img:
        # 裁剪图片以保持宽高比
        img = img.crop((0, center_height, width, center_height + width))  # 裁剪图像
        # 调整图片大小，使用 LANCZOS 进行抗锯齿处理
        img = img.resize((img_dim, img_dim), Image.Resampling.LANCZOS)  # 调整大小
        # 转换为 NumPy 数组并归一化
        img = np.array(img).astype(np.float32) / 255 * 2 - 1  # 归一化到 [-1, 1]
    return img

# 示例使用
image_data = imread(imgs[0])


# imgs = glob.glob('img_align_celeba/*.jpg')
# np.random.shuffle(imgs)

# height,width = misc.imread(imgs[0]).shape[:2]
# center_height = int((height - width) / 2)
# img_dim = 64
# z_dim = 512


# def imread(f):
#     x = misc.imread(f)
#     x = x[center_height:center_height+width, :]
#     x = misc.imresize(x, (img_dim, img_dim))
#     return x.astype(np.float32) / 255 * 2 - 1


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X,None
                X = []


x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in
x = Conv2D(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)

encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]

# 解码层，也就是生成器部分
z_in = Input(shape=K.int_shape(x)[1:])
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)
z = Conv2DTranspose(z_dim/2, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/4, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/8, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(z_dim/16, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = Activation('tanh')(z)

decoder = Model(z_in, z)
decoder.summary()

class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, shift, log_scale = inputs
        z = K.exp(log_scale) * z + shift
        logdet = -K.sum(K.mean(log_scale, 0))
        self.add_loss(logdet)
        return z

z_shift = Dense(z_dim)(x)
z_log_scale = Dense(z_dim)(x)
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])

recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss

vae = Model(x_in, x_out)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-4))


def sample(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3),dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            #生成样本
            x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))
            digit = x_recon[0]
           
            # 确保 digit 是 (img_dim, img_dim, 3) 并且转换为 uint8
            if digit.shape[-1] == 1:  # 如果是灰度图像
                digit = np.repeat(digit, 3, axis=-1)
            
            digit = np.clip((digit + 1) / 2 * 255, 0, 255).astype(np.uint8)

            # 放置到 figure 中
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    # figure = (figure + 1) / 2 * 255
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('samples'):
            os.mkdir('samples')
    def on_epoch_end(self, epoch, logs=None):
        path = 'samples/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')


evaluator = Evaluate()

vae.fit_generator(data_generator(),
                  epochs=5,#1000,
                  steps_per_epoch=1000,
                  callbacks=[evaluator])
