"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks
"""
# 导入PyTorch的相关模块
import torch
from torch import nn
from torch.nn import functional as F
from networks.Layers import *  # 导入自定义的Layer模块

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()
    
    # q(y|x)：y的条件概率密度，通过一系列线性层和ReLU激活函数处理
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),  # 将输入维度映射到512
        nn.ReLU(),              # 应用ReLU激活函数
        nn.Linear(512, 512),    # 再次映射到512
        nn.ReLU(),              # 应用ReLU激活函数
        GumbelSoftmax(512, y_dim)   # logits概率的y
    ])

    # q(z|y,x)：z的条件概率密度，输入为x和y的组合，通过线性层和ReLU激活函数处理
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),  # 输入维度为x和y的组合映射到512
        nn.ReLU(),                       # 应用ReLU激活函数
        nn.Linear(512, 512),            # 再次映射到512
        nn.ReLU(),                       # 应用ReLU激活函数
        Gaussian(512, z_dim)            # mu, var, z
    ])

  # q(y|x)：计算y的条件概率密度
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        # 最后一层是gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)：计算z的条件概率密度，输入为x和y的组合
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  # 将x和y沿维度1合并
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, temperature=1.0, hard=0):
    # x = Flatten(x)

    # q(y|x)：计算y的条件概率密度
    logits, prob, y = self.qyx(x, temperature, hard)
    
    # q(z|x,y)：计算z的条件概率密度
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output


# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)：z的边缘概率密度，y的条件概率密度，通过线性层和ReLU激活函数处理
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)：x的条件概率密度，输入为z，通过多个线性层和ReLU激活函数处理，最后应用Sigmoid函数将输出值归一化
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),   # 将z映射到512
        nn.ReLU(),              # 应用ReLU激活函数
        nn.Linear(512, 512),    # 再次映射到512
        nn.ReLU(),              # 应用ReLU激活函数
        nn.Linear(512, x_dim),  # 映射到x维度
        torch.nn.Sigmoid()      # 应用Sigmoid函数进行归一化
    ])

  # p(z|y)：计算z的边缘概率密度
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))  # 使用softplus函数处理variance
    return y_mu, y_var
  
  # p(x|z)：计算x的条件概率密度
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)：计算z的边缘概率密度
    y_mu, y_var = self.pzy(y)
    
    # p(x|z)：计算x的条件概率密度