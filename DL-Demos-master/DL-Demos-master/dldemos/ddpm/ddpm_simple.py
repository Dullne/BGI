import torch


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        print(betas.shape)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    # 前向采样函数，用于根据给定参数和噪声生成采样结果
    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res
    # 后向采样函数，用于根据给定参数和模型生成采样结果
    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        # 在sample_backward_step中，我们先准备好这一步的神经网络输出eps。为此，我们要把整型的t转换成一个格式正确的Tensor。考虑到输入里可能有多个batch，我们先获取batch size n，再根据它来生成t_tensor。
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor) #调用的net的forward

        # 根据伪代码，我们仅在t非零的时候算方差项。方差项用到的方差有两种取值，效果差不多，我们用simple_var来控制选哪种取值方式。获取方差后，我们再随机采样一个噪声，根据公式，得到方差项。
        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)
        # 最后，我们把eps和方差项套入公式，得到这一步更新过后的图像x_t。
        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t
