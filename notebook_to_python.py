# %%
# Basic Math
print("hello, this is notebook")

# %%
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
from torch import nn
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, TensorDataset


# %%
s_curve, _ = make_s_curve(10000, noise=0.1)
print(s_curve.shape)
s_curve = s_curve[:, [0, 2]]/10.0 # 只保留x和z坐标，并缩放到0-1范围内
print("shape 2D:",s_curve.shape)
# plt.scatter(s_curve[:, 0], s_curve[:, 1], s=10, color="orange")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("S-curve")
# plt.show()
print("S曲线是若干离散点组成，看作是一种数据分布")

# %%
data = torch.from_numpy(s_curve).float()
print("dataset shape:", data.shape)

# DDMP
noise_steps = 200
betas = torch.linspace(1e-4, 0.02, noise_steps)
# betas = torch.linspace(-6, 6, noise_steps)
# betas = betas.sigmoid() * (5e-3 - 1e-5) + 1e-5 # sigmoid平滑过渡
alphas = 1.0 - betas
print("betas shape:", betas.shape)

#
alphas_bar = torch.cumprod(alphas, dim=0) # 累积乘积，得到每一步的alpha_bar
alphas_bar_sqrt = torch.sqrt(alphas_bar) # alpha_bar的平方根
one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - alphas_bar) # 1 - alpha_bar的平方根

print("betas shape:", betas.shape)
print("alphas prod", alphas_bar[0], alphas_bar[-1])

# plot
plt.plot(alphas_bar.numpy(), label="alpha_bar")
plt.plot(alphas_bar_sqrt.numpy(), label="alpha_bar_sqrt")
plt.legend()
plt.show()

# %%
# 前向扩散过程 q(x_t | x_0) 
# q(x_t|x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
class Diffusion:
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        从前向扩散过程中采样 xt
        x0: 原始数据，shape (npts, dim)
        t: 时间步，shape (n, 1) 多个时间步
        返回 xt，shape (npts, dim)，
             以及对应的噪声 epsilon，shape (npts, dim)
        """
        # 获取对应时间步的alpha_bar和1-alpha_bar的平方根
        alpha_bar_sqrt_t = alphas_bar_sqrt[t].unsqueeze(-1) # shape (n, 1)
        one_minus_alpha_bar_sqrt_t = one_minus_alphas_bar_sqrt[t].unsqueeze(-1) # shape (n, 1)
        # 从标准正态分布中采样噪声
        noise = torch.randn_like(x0) # shape (npts, dim)
        # 计算x_t
        xt = alpha_bar_sqrt_t * x0 + one_minus_alpha_bar_sqrt_t * noise # shape (npts, dim)
        #
        return xt, noise

# %%
# 测试前向扩散过程
"""
diffusion = Diffusion()
# 不同时间步
xt_data = []
epsilon_data = []
t_data = []
num_samples = 100 # 总共抽取num_samples个样本作为训练数据集, 决定数据分布
sample_npts = 256 # 每个样本抽取 batch_size个随机坐标点作为当前样本
batch_t = noise_steps // 3 # 抽取 30%时间步，而非单一个时间步，增加训练数据的多样性
for ns in range(num_samples):
    perm = torch.randperm(dataset.shape[0])[:sample_npts] # 从dataset中随机抽取sample_npts个坐标点
    x0 = dataset[perm] # shape (sample_npts, 2)
    ts = torch.randint(0, noise_steps, size=(batch_t,)) # 随机抽取batch_t时间步
    xt, epsilon_true = diffusion.q_sample(x0, ts) # 测试不同时间步的采样结果
    if ns % 20 == 0: # 每n步打印一次
        print("xt shape:", xt.shape)
        # plt.scatter(xt[0, :, 0].numpy(), xt[0, :, 1].numpy(), s=10, color="red")
        # plt.title(f"num sample idx {ns}")
        # plt.show()
    xt_data.append(xt)
    epsilon_data.append(epsilon_true)
    t_data.append(ts.unsqueeze(1)) # shape (batch_t, 1)

# list [steps, n, 2] stack to tensor
xt_data = torch.cat(xt_data, dim=0) # shape (nsample*batch_t, npts, 2)
epsilon_data = torch.cat(epsilon_data, dim=0) # shape (nsample*batch_t, npts, 2)
t_data = torch.cat(t_data, dim=0) # shape (nsample*batch_t, 1)
print("xt_data shape:", xt_data.shape)
print("t_data shape:", t_data.shape)
"""
# %%
# 1.搭建噪声学习网络epsilon_theta
# 2. 数据学习训练 
#    loss = E[||epsilon - epsilon_theta(x_t, t)||^2]

# 可以选择2种网络架构
# （1）UNet 结构的噪声预测网络
# （2）Transformer 结构的噪声预测网络
class NoisePredictorNet(nn.Module):
    # 此处仅为演示使用简单net结构 MLP
    def __init__(self, num_steps:int, num_units:int=128, num_emding:int=16):
        super(NoisePredictorNet, self).__init__()

        # 时间步嵌入层
        self.embeding = nn.Embedding(num_steps, num_emding)

        # 线性层网络, 串联
        # xt 图片 + time embding
        self.linear = nn.Sequential(
                nn.Linear(2+num_emding, num_units),
                nn.SiLU(),
                nn.Linear(num_units, num_units),
                nn.SiLU(),
                nn.Linear(num_units, num_units),
                nn.SiLU(),
                nn.Linear(num_units, 2)
        )


    # def qprocess(self, x0:torch.Tensor, t:torch.Tensor):
    #     """前向扩散过程：在t时刻给x0添加噪声"""
    #     noise = torch.randn_like(x0)
    #     a = alphas_bar_sqrt[t].unsqueeze(1)  # 适配batch维度
    #     b = one_minus_alphas_bar_sqrt[t].unsqueeze(1)
    #     xt = a * x0 + b * noise
    #     return xt, noise
    
    # def p_process(self, xt:torch.Tensor, t:torch.Tensor):
    #     """反向扩散过程：从xt去噪一步"""
    #     # 获取当前时间步的参数
    #     beta_t = betas[t].unsqueeze(1)
    #     alpha_t = alphas[t].unsqueeze(1)
    #     coef = one_minus_alphas_bar_sqrt[t].unsqueeze(1)
        
    #     # 拼接时间步特征
    #     t_emb = t / noise_steps
    #     t_emb = t_emb.unsqueeze(1)
    #     xt_input = torch.cat([xt, t_emb], dim=1)
        
    #     # 预测噪声
    #     noise_pred = self.linear(xt_input)
        
    #     # 计算去噪后的样本
    #     x_prev_mean = (1 / torch.sqrt(alpha_t)) * (
    #         xt - (beta_t / coef) * noise_pred
    #     )
        
    #     # 只有在最后一步不加噪声
    #     if t[0] == 0:
    #         x_prev = x_prev_mean
    #     else:
    #         # 添加少量噪声保持扩散特性
    #         noise = torch.randn_like(xt)
    #         x_prev = x_prev_mean + torch.sqrt(beta_t) * noise
        
    #     return x_prev
              

    def forward(self, xt:torch.Tensor, t:torch.Tensor):
        """
        xt: shape (npts, 2)
        t: shape (n, 1)
        返回 epsilon_theta(xt, t)，shape (npts, 2)
        """

        # 前向扩散生成带噪数据
        # xt, noise = self.qprocess(x0, t)

        if t.dim() > 1:
            # t维度到 (npts, 1)
            t = t.unsqueeze(-1)

        t_embeding = self.embeding(t) # 时间步嵌入，shape (n, num_units)
        # t_emb = t / noise_steps
        # t_embeding = t_emb.unsqueeze(1)
        # 拼接时间步嵌入和输入
        xt_cat =  torch.cat([xt, t_embeding], dim=-1) # shape (npts, 2 + num_units)
        #最后一层线性层输出噪声预测 epsilon_theta
        epsilon_theta_mu = self.linear(xt_cat) # 线性层 (npts, 2)
        return epsilon_theta_mu


# %%
#  训练噪声学习网络
# 学习 epsilon_theta(xt, t) 来预测噪声
# loss 为 MSE loss = E[||epsilon - epsilon_theta(x_t, t)||^2]
def loss_calculator(epsilon_pred: torch.Tensor, epsilon_true: torch.Tensor) -> torch.Tensor:
    return (epsilon_pred - epsilon_true).square().mean()

# 训练循环
epochs = 800
sample_npts = 256

qprocess = Diffusion()
mlpnet = NoisePredictorNet(num_steps=noise_steps)
optimizer = Adam(mlpnet.parameters(), lr=1e-3)
loss_history = []

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=sample_npts, shuffle=True)

for epoch in range(epochs):
    # 训练过程中每个epoch都要遍历整个数据集
    t = torch.randint(0, noise_steps, (sample_npts,))
    # # batch个xt图像数据
    perm = torch.randperm(data.shape[0])
    
    ## 非常重要epoch内循环
    # 保证每个epoch都能看到所有数据，且每个batch随机抽取不同时间步t
    for start in range(0, perm.shape[0], sample_npts):
        x0 = data[start:start+sample_npts] # shape (npts, dim)
        
        # 随机采样时间步
        t = torch.randint(0, noise_steps, (x0.shape[0],))

        # 从前向扩散过程中采样 xt 和对应的噪声 epsilon
        xt, noise = qprocess.q_sample(x0, t) # shape (npts, dim), shape (npts, dim)

        # 模型预测当前xt，t的噪声
        epsilon_pred = mlpnet(xt, t) # [npts, dim]

        # 计算预测和真实噪声的loss
        cur_loss = loss_calculator(noise, epsilon_pred)
        # 反向传播计算
        optimizer.zero_grad() # grad 清零
        cur_loss.backward()
        # 根据loss更新模型
        optimizer.step()
        ##
        loss_history.append(cur_loss.item())
        if len(loss_history) > 10:
            loss_history.append(np.mean(loss_history))

    if epoch % 50 == 0:
        print(f" epoch {epoch}/{epochs}, loss mean {loss_history[-1]:.4f}")

torch.save(mlpnet.state_dict(), "mlpnet_state_dict.pt")
plt.plot(loss_history)
plt.title(f"loss history of {epochs} epochs")
plt.show()


# %%
# 逆扩散采样
# p(x_t-1|x_t) = N(mu_t, I) + sigma_t*z
# x_t-1 = 1/sqrt(alpha_t)(xt - beta_t/sqrt(1-alpha_bar_t)*epsilon_theta(xt, t)) + sigma_t*z
# sigma_t assumpted to be sqrt(beta_t)
def sampling(xt: torch.Tensor, t: torch.Tensor)->torch.Tensor:
    """
    逆扩散采样
    t: 当前时间步，shape (n, 1), 所有t都一样的
    xt: 当前图像，shape (npts, dim)
    """
    coeff = betas[t] / (one_minus_alphas_bar_sqrt[t] + 1e-15)
    coeff = coeff.unsqueeze(-1) # shape (n, 1)
    epsilon_pred = mlpnet(xt, t) # shape (npts, 2)
    rep = 1.0/(alphas[t].sqrt().unsqueeze(-1)+1e-15) # shape (n, 1)
    
    # 均值路径
    mean_walk = rep*(xt - coeff*epsilon_pred)

    # 随机路径
    z = torch.randn_like(xt) # 标准正态分布噪声
    rnd_walk = torch.sqrt(betas[t]).unsqueeze(-1)* z

    # 恢复图像
    if t[0] > 0:
        xtm1 = mean_walk + rnd_walk
    else:
        # 最后一步不加噪声
        xtm1 = mean_walk
    
    return xtm1



def plot_tensor_image(xt: torch.Tensor, ti, steps):
    plt.scatter(xt[:, 0].detach().numpy(), xt[:, 1].detach().numpy(), s=5, color="green")
    plt.title(f" time {ti}/{steps}")
    plt.show()


# 高斯分布
npts, dim = data.shape
# npts, dim = sample_npts, 2
xT = torch.randn((npts, dim)) # 从标准正态分布初始化
# plot_tensor_image(xT, noise_steps, noise_steps)

torch.no_grad() # 采样过程中不需要计算梯度
mlpnet.eval() # 采样过程中不需要更新模型参数

# 逐步恢复 from t=T to t = 0
xt = xT
for ti in reversed(range(noise_steps)):
    t_tensor = torch.full((npts,), ti)
    xt = sampling(xt, t_tensor) # shape (npts, 2)
    # xt = mlpnet.p_process(xt, t_tensor) # shape (npts, 2)

    if ti % 100 == 0:
        print (f" current time{ti}/{noise_steps}")
        plot_tensor_image(xt, ti, noise_steps)

print(" inferred the last x0")
plot_tensor_image(xt, ti, noise_steps)

