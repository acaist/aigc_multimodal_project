import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_s_curve


# 设置随机种子确保可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成circle数据集
def generate_circle(n_samples=10000):
    """生成S型曲线数据"""
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = np.sin(t)
    y = np.cos(t)
    
    # 调整成S型并添加少量噪声
    x = x * 2
    y = np.where(t < np.pi, y, -y) * 1.5
    
    # 添加轻微噪声让数据更真实
    x += np.random.normal(0, 0.05, n_samples)
    y += np.random.normal(0, 0.05, n_samples)
    
    # 转换为tensor
    data = np.stack([x, y], axis=1)
    return torch.tensor(data, dtype=torch.float32)

def generate_s_curve(n_samples=10000):
    s_curve, _ = make_s_curve(n_samples, noise=0.1)
    print(s_curve.shape)
    s_curve = s_curve[:, [0, 2]]/10.0 # 只保留x和z坐标，并缩放到0-1范围内
    print("shape 2D:",s_curve.shape)
    return torch.tensor(s_curve, dtype=torch.float32)

# 2. 实现DDPM核心类
class SimpleDDPM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_steps=100):
        super().__init__()
        self.n_steps = n_steps
        self.input_dim = input_dim
        
        # 预计算前向扩散的参数
        self.beta = torch.linspace(1e-4, 0.02, n_steps)  # 噪声调度
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # 累积乘积
        
        # 噪声预测网络 (简单的MLP)
        self.noise_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # 输入：数据(2维) + 时间步(1维)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)      # 输出：预测的噪声(2维)
        )

        # plot
        plt.plot(self.alpha_bar.numpy(), label="alpha_bar")
        plt.legend()
        plt.show()
    
    def forward_diffusion(self, x0, t, noise=None):
        """前向扩散过程：在t时刻给x0添加噪声"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        alpha_bar_t = self.alpha_bar[t].unsqueeze(1)  # 适配batch维度
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise
    
    def forward(self, x0, t):
        """前向传播：预测噪声"""
        # 前向扩散生成带噪数据
        xt, noise = self.forward_diffusion(x0, t)
        
        # 拼接时间步特征
        t_emb = t / self.n_steps  # 归一化时间步到[0,1]
        t_emb = t_emb.unsqueeze(1)
        xt_input = torch.cat([xt, t_emb], dim=1)
        
        # MLP预测噪声
        noise_pred = self.noise_mlp(xt_input)
        return noise_pred, noise
    
    def reverse_diffusion(self, xt, t):
        """反向扩散过程：从xt去噪一步"""
        # 获取当前时间步的参数
        beta_t = self.beta[t].unsqueeze(1)
        alpha_t = self.alpha[t].unsqueeze(1)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(1)
        
        # 拼接时间步特征
        t_emb = t / self.n_steps
        t_emb = t_emb.unsqueeze(1)
        xt_input = torch.cat([xt, t_emb], dim=1)
        
        # 预测噪声
        noise_pred = self.noise_mlp(xt_input)
        
        # 计算去噪后的样本
        x_prev_mean = (1 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        
        # 只有在最后一步不加噪声
        if t[0] == 0:
            x_prev = x_prev_mean
        else:
            # 添加少量噪声保持扩散特性
            noise = torch.randn_like(xt)
            x_prev = x_prev_mean + torch.sqrt(beta_t) * noise
        
        return x_prev
    
    def sample(self, n_samples):
        """从纯噪声开始生成样本"""
        # 从标准正态分布初始化
        xt = torch.randn(n_samples, self.input_dim)
        
        # 逐步去噪
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long)
            xt = self.reverse_diffusion(xt, t_tensor)
        
        return xt

# 3. 训练和可视化
def train_and_visualize():
    # 生成数据集
    data = generate_s_curve(n_samples=10000)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = SimpleDDPM(input_dim=2, hidden_dim=128, n_steps=100).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(ddpm.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 训练模型
    n_epochs = 800
    ddpm.train()
    for epoch in tqdm(range(n_epochs), desc="Training DDPM"):
        total_loss = 0
        for batch in dataloader:
            x0 = batch[0].to(device)
            batch_size = x0.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, ddpm.n_steps, (batch_size,), device=device)
            
            # 前向传播
            noise_pred, noise_true = ddpm(x0, t)
            
            # 计算损失并优化
            loss = criterion(noise_pred, noise_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每20个epoch打印一次损失
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
    
    # 生成样本
    ddpm.eval()
    with torch.no_grad():
        generated_samples = ddpm.sample(n_samples=10000).cpu().numpy()
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    original_data = data.numpy()
    plt.scatter(original_data[:, 0], original_data[:, 1], s=1, alpha=0.5)
    plt.title('Original S Curve Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)
    
    # 生成的数据
    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=1, alpha=0.5, c='orange')
    plt.title('Generated S Curve Data (DDPM + MLP)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()

# 运行训练和可视化
if __name__ == "__main__":
    train_and_visualize()