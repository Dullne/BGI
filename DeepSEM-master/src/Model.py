import numpy as np
import torch
import torch.nn.functional  as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter

# Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor

# 定义KL散度损失函数
def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


# We followed implement in https://github.com/jariasf/GMVAE/tree/master/pytorch
class LossFunctions:
    eps = 1e-8
    # 重构损失函数
    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none').mean()
        else:
            raise Exception  # 如果指定了未知的损失类型，抛出异常
        return loss
    # 计算正态分布的对数概率密度函数
    def log_normal(self, x, mu, var):
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            # torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)
            torch.log(torch.FloatTensor([2.0 * np.pi])).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)
    # 高斯分布的损失函数   后验分布-先验分布
    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()
    # 熵损失函数
    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

# Gumbel-Softmax层
class GumbelSoftmax(nn.Module):
    #                     128    1 
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, ):
        y = self.gumbel_softmax_sample(logits, temperature)
        return y

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.c_dim)  #张量的第一个维度的大小将由第二个维度的大小决定，而第二个维度的大小由 self.c_dim 决定。
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)  #z_dim=1
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        print("推断网络的输出：")
        print(mu.squeeze(2).shape,logvar.squeeze(2).shape)
        return mu.squeeze(2), logvar.squeeze(2)


class InferenceNet(nn.Module):
    #                      1   128    1      910     nn.Tanh()
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
        self.inference_qyx = torch.nn.ModuleList([
            nn.Linear(n_gene, z_dim),
            nonLinear,  #Tanh()
            nn.Linear(z_dim, z_dim),
            nonLinear,
            GumbelSoftmax(z_dim, y_dim)
        ])  #i: 910  o: 3
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            Gaussian(z_dim, 1)
        ])  #i: 2   o: 

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std

        return z

    def qyx(self, x, temperature):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def qzxy(self, x, y):
        concat = torch.cat((x, y.unsqueeze(1).repeat(1, x.shape[1], 1)), dim=2)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, adj, temperature=1.0):
        logits, prob, y = self.qyx(x.squeeze(2), temperature)
        mu, logvar = self.qzxy(x, y)
        mu_ori = mu
        mu = torch.matmul(mu, adj)
        logvar = torch.matmul(logvar, adj)
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        output = {'mean'  : mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y, 'mu_ori': mu_ori}
        # print("mu:",mu.shape,"var:",var.shape,"z:",z.shape,"y:",y.shape)
        return output


class GenerativeNet(nn.Module):
     #                   1     128    1       910 nn.Tanh()
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene
        self.y_mu = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))
        self.y_var = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))

        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(1, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_logvar = self.y_var(y)
        return y_mu, y_logvar

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y, adj):
        y_mu, y_logvar = self.pzy(y)
        y_mu = torch.matmul(y_mu, adj)
        y_logvar = torch.matmul(y_logvar, adj)
        y_var = torch.exp(y_logvar)
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        print("y_mu:",y_mu.shape,"y_logvar:",y_var.shape,"x_rec:",x_rec.shape)
        output = {'y_mean': y_mu.view(-1, self.n_gene), 'y_var': y_var.view(-1, self.n_gene), 'x_rec': x_rec}
        return output


class VAE_EAD(nn.Module):
    #                   910*910   1   128    1
    def __init__(self, adj_A, x_dim, z_dim, y_dim,):
        super(VAE_EAD, self).__init__()
        #adj_A作为一个nn.Parameter，确实可以被模型自动训练，但这是在损失函数中包含这个参数作为可训练变量的情况下。
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True, name='adj_A'))
        self.n_gene = n_gene = len(adj_A)
        nonLinear = nn.Tanh()
        self.inference = InferenceNet(x_dim, z_dim, y_dim, n_gene, nonLinear) # 1  128  1   910 nn.Tanh()
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, n_gene, nonLinear)  # 1  128  1   910 nn.Tanh()
        self.losses = LossFunctions()
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)
        
    def _one_minus_A_t(self, adj):
        adj_normalized = Tensor(np.eye(adj.shape[0])) - (adj.transpose(0, 1))
        return adj_normalized

    def forward(self, x, dropout_mask, temperature=1.0, opt=None, ):
        x_ori = x
        x = x.view(x.size(0), -1, 1) # (64*910*1)
        mask = Variable(torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(), requires_grad=False)#.cuda()
        adj_A_t = self._one_minus_A_t(self.adj_A * mask) #GRN layer  (I-W^T)
        print("GRN layer:",adj_A_t.shape)
        adj_A_t_inv = torch.inverse(adj_A_t)  #逆 GRN layer  (I-W^T)^-1
        #推断网络
        out_inf = self.inference(x, adj_A_t, temperature)
        z, y = out_inf['gaussian'], out_inf['categorical']
        #生成网络
        z_inv = torch.matmul(z, adj_A_t_inv)
        out_gen = self.generative(z_inv, y, adj_A_t)

        # writer = SummaryWriter("graph_logs")
        # writer.add_graph(self.generative, (Tensor(z_inv), Tensor(y), Tensor(adj_A_t)))
        # writer.close()

        output = out_inf
        for key, value in out_gen.items():
            output[key] = value  #整合推断和生成网络的输出
        dec = output['x_rec']
        #重构损失
        loss_rec = self.losses.reconstruction_loss(x_ori, output['x_rec'], dropout_mask, 'mse')
        #kl损失  先验分布和后验分布的差别
        loss_gauss = self.losses.gaussian_loss(z, output['mean'], output['var'], output['y_mean'], output['y_var']) * opt.beta
        #分类损失
        loss_cat = (-self.losses.entropy(output['logits'], output['prob_cat']) - np.log(0.1)) * opt.beta
        #loss_guass+loss_cat=kl
        loss = loss_rec + loss_gauss + loss_cat
        return loss, loss_rec, loss_gauss, loss_cat, dec, y, output['mean']







