import torch

# 定义一个DDPM类，用于模拟深度正则化概率模型
class DDPM():
    # 初始化函数，参数包括设备、步数、最小beta和最大beta，默认beta值
    def __init__(self,
                 device,  # 计算设备
                 n_steps: int,  # 步数
                 min_beta: float = 0.0001,  # 最小beta值，默认0.0001
                 max_beta: float = 0.02):  # 最大beta值，默认0.02
        # 生成beta值，从最小beta到最大beta，步数为n_steps，并移动到指定设备
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        # 计算alpha值，等于1减去beta值
        alphas = 1 - betas
        # 创建一个与alphas形状相同的tensor，用于存储alpha_bar值
        alpha_bars = torch.empty_like(alphas)
        # 初始化product值为1
        product = 1
        # 遍历alphas值，计算alpha_bar值，即累计乘积
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        # 将beta值、步数、alpha值和alpha_bar值赋值给实例变量
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    # 前向采样函数，用于根据给定参数和噪声生成采样结果
    def sample_forward(self, x, t, eps=None):
        # 获取alpha_bar值，并调整其形状以便进行元素相乘
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        # 如果eps为None，则生成与x形状相同的随机噪声
        if eps is None:
            eps = torch.randn_like(x)
        # 计算采样结果，eps * sqrt(1 - alpha_bar) + sqrt(alpha_bar) * x
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        # 返回采样结果
        return res

    # 后向采样函数，用于根据给定参数和模型生成采样结果
    def sample_backward(self, img_shape, net, device, simple_var=True):
        # 生成形状为img_shape的随机噪声，并移动到指定设备
        x = torch.randn(img_shape).to(device)
        # 将网络移动到指定设备
        net = net.to(device)
        # 从倒数第二步到第一步进行迭代，反向采样
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        # 返回最终的采样结果
        return x

    # 后向采样步骤函数，用于单步后向采样
    def sample_backward_step(self, x_t, t, net, simple_var=True):
        # 获取x_t的批量大小
        n = x_t.shape[0]
        # 创建形状为[n, 1]的tensor，用于存储t值，并将其移动到x_t的设备
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        # 使用网络对当前x_t和t进行采样，得到eps值
        eps = net(x_t, t_tensor)

        # 如果t为0，则设置noise为0
        if t == 0:
            noise = 0
        else:
            # 如果simple_var为True，则直接使用beta值作为方差
            if simple_var:
                var = self.betas[t]
            else:
                # 否则，计算方差，使用alpha_bar的前一个值与当前值计算
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            # 生成与x_t形状相同的随机噪声
            noise = torch.randn_like(x_t)
            # 将噪声缩放为方差大小
            noise *= torch.sqrt(var)

        # 计算均值，使用x_t和eps进行计算
        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])
        # 更新x_t，使用均值和噪声
        x_t = mean + noise

        # 返回更新后的x_t
        return x_t