定义一个连乘 $\bar\alpha_t :=\prod_{t=1}^T{\alpha_t}$  

$q(X_t|X_0) = N(X_t; \sqrt{\bar \alpha_t}X_0, (1-\bar\alpha_t)I)$

$q(X_{t-1}|X_0) = N(X_t; \sqrt{\bar \alpha_{t-1}}X_0, (1-\bar\alpha_{t-1})I)$

$q(X_t|X_{t-1}, X_0) = q(X_t|X_{t-1}) = N(X_t; \sqrt{\alpha_t}X_{t-1}, (1-\alpha_t)I)$  

所以： $q(X_t|X_{t-1}, X_0) *q(X_{t-1}|X_0)/q(X_t|X_0)$
$$
\begin{align}
& \propto \exp(\frac{(x-\sqrt{\bar\alpha_t}X_{t-1})^2}{2(1-\alpha_t)I} + \frac{(x-\sqrt{\bar\alpha_{t-1}}X_0)^2}{2(1-\sqrt{\alpha_{t-1}})I} - \frac{(x-\sqrt{\bar\alpha_{t}}X_0)^2}{2(1-\sqrt{\alpha_{t}})I})  \\
& 因为X_t是X_0加标准高斯噪声z得到，X_t = \sqrt{\bar\alpha_t}X_0 + \sqrt{1-\bar\alpha_t}z \\
& 可以改写一下 X_0 = \frac{X_t - \sqrt{1-\bar\alpha_t}z}{\sqrt{\bar\alpha_t}} 带入上面公式消去X_0\\
& \propto \exp(\frac{(x-\sqrt{\bar\alpha_t}X_{t-1})^2}{2(1-\alpha_t)I} + \frac{(x-\sqrt{\bar\alpha_{t-1}/\bar\alpha_t}(X_t - \sqrt{1-\bar\alpha_t}z))^2}{2(1-\sqrt{\alpha_{t-1}})I} - \frac{(x-(X_t - \sqrt{1-\bar\alpha_t}z))^2}{2(1-\sqrt{\alpha_{t}})I})\\
&上述公式是X_{t-1}的二次方程，\\
& 而最后使用凑方法凑成(x+b/2a)^2 +C
\end{align}
$$
- 最后化简可以推出其分布的均值$\bar\mu(X_t, t) = \frac{1}{\sqrt{\bar\alpha_t}}(X_t -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}z_t)$
- 网络$p_\theta$要学习的就是这个均值，尽可能KL散度小，注意到均值有个随机分布$z_t \in N(0, I)$

---
$p_\theta(X_0)$是概率密度函数，是经过反向去噪网络生成的样本$X_0$的似然概率likelihood，反向过程马尔科夫链的联合概率分布的边缘分布, 即
$p_\theta(X_0) := \int p_\theta(X_{0:T}dx_{1:T})$。  
学习的目的就是让$p_\theta(x_0)$接近真实$q(X_0)$。我们最大化似然即最大化$\log p_\theta(X_0)$对数似然分布。  
为了便于优化，我们通常不直接计算 $p_\theta(X_0)$，而是使用一个可计算的变分下界（Evidence Lower Bound, ELBO）​ 来替代。这个下界由一系列在时间步 t 上的项组成：
$$
-\log p_\theta(X_0) \le -\log p_\theta(X_0) + \log D_{KL}(q(X_{1:T}|X_0)||p_\theta(X_{1:T}|X_0))\\
\le -\log p_\theta(X_0) + \mathbb E_{X_{1:T} \in q(X_{1:T}|X_0)}[\log \frac{q(X_{1:T}|X_0)}{p_\theta(X_{0:T})/p_\theta(X_0)}]\\
=-\log p_\theta(X_0) + \mathbb E_{X_{1:T} \in q(X_{1:T}|X_0)}[\log \frac{q(X_{1:T}|X_{0})}{p_\theta(X_{0:T})} + \log p_\theta(X_0)] \\
= \mathbb E_{X_{1:T} \in q(X_{1:T}|X_0)}[\log \frac{q(X_{1:T}|X_{0})}{p_\theta(X_{0:T})}] \\
$$  
上述$\mathbb E_{X_{1:T} \in q(X_{1:T}|X_0)}[\log \frac{q(X_{1:T}|X_{0})}{p_\theta(X_{0:T})}]$ 即是 $-log p_\theta(X_0)$的上界，最小化其上界即最小化负对数的似然概率。进一步可以化简该上界：  
$$
\mathbb E_{X_{1:T} \in q(X_{1:T}|X_0)}[\log \frac{q(X_{1:T}|X_{0})}{p_\theta(X_{0:T})}] = \\
\mathbb E_q[\log \frac{q(X_T|X_{T-1})*q(X_{T-1}|X_{T-2})*...q(X_1|X_0)}{p_\theta(X_T)*p_\theta(X_{T-1}|X_{T})*...*p_\theta(X_1|X_2)*p_\theta(X_0|X_1)}] \\
=\mathbb E_q[-\log p_\theta(X_T) + \log \frac{q(X_T|X_{T-1})...q(X_2|X_1))}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)} + \log \frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}] \\
注意之前推导过的q(X_{T-1}|X_T, X_0)=q(X_T|X_{T-1}, X_0) *q(X_{T-1}|X_0)/q(X_T|X_0) \\
则q(X_T|X_{T-1}, X_0)=q(X_{T-1}|X_T, X_0)/q(X_{T-1}|X_0)*q(X_T|X_0) \\
q是扩散过程，是从0->T的过程q(X_T|X_{T-1}),而加入条件反扩散q(X_{T-1}|X_T, X_0)则需要凑一个概率q(X_{T-1}|X_0)/q(X_T|X_0)\\
带入上式得：
=\mathbb E_q[-\log p_\theta(X_T) + \log \frac{q(X_{T-1}|X_T,X_0)...q(X_1|X_2,X_0))}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)}*\frac{q(X_T|X_0)...q(X_2|X_0)}{q(X_{T-1}|X_0)...q(X_1|X_0)} + \log \frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}]\\
=\mathbb E_q[-\log p_\theta(X_T) + \log \frac{q(X_{T-1}|X_T,X_0)...q(X_1|X_2,X_0))}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)} + \log \frac{q(X_T|X_0)...q(X_2|X_0)}{q(X_{T-1}|X_0)...q(X_1|X_0)} + \log \frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}] \\
=\mathbb E_q[-\log p_\theta(X_T) + \log \frac{q(X_{T-1}|X_T,X_0)...q(X_1|X_2,X_0))}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)} + \log \frac{q(X_T|X_0)}{q(X_1|X_0)} + \log \frac{q(X_1|X_0)}{p_\theta(X_0|X_1)}]
\\
=\mathbb E_q[\log \frac{q(X_T|X_0)}{p_\theta(X_T)} + \log \frac{q(X_{T-1}|X_T,X_0)...q(X_1|X_2,X_0))}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)} - \log {p_\theta(X_0|X_1)}] \\
最后归类为3项：
（1）L_T = \mathbb E_q\log \frac{q(X_T|X_0)}{p_\theta(X_T)}, 即D_{KL}(q(X_T|X_0)||p_\theta(X_T)),X_T是标准高斯分布的，不含学习参数，学习时候可以忽略此项\\
(2) L_{T-1} =\mathbb E_q \log \frac{q(X_{T-1}|X_T,X_0)...q(X_1|X_2,X_0))}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)} 即扩散过程q与逆扩散过程p_\theta的KL散度\\
(3) L_0 = \mathbb E_q[-\log {p_\theta(X_0|X_1)}],即初始时刻，可与（2）一同学习训练\\
$$

论文将$p_\theta(X_{T-1}|X_T)$的方差设置为常数，$p_\theta$对均值$\bar\mu$学习，
$$
L_{T-1} = \mathbb E_q \log \frac{q(X_{T-1}|X_T,X_0)...q(X_1|X_2,X_0))*q(X_0|X_1, X_0)}{p_\theta(X_{T-1}|X_T)...p_\theta(X_1|X_2)p_\theta(X_0|X_1)} \\
$$
q和$p_\theta$都是高斯分布，则其KL散度,即2个高斯分布  
$L_{T-1} = \mathbb E_q[\frac{1}{2\sigma_t^2}(|\bar\mu_t(X_t, X_0) - \mu_\theta(X_t, t)|^2)] + C$  
即学习扩散过程的均值$\bar\mu_t(X_t, X_0)$即可,根据上文推导  
$\bar\mu_t(X_t, t) = \frac{1}{\sqrt{\bar\alpha_t}}(X_t -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}z_t)$  
注意$X_t$是$X_0$和随机噪声$z$的函数 $X_t = \sqrt{\bar\alpha_t}X_0 + \sqrt{1-\bar\alpha_t}z$  
即令均值形式相同，引入一个可学习的网络$\epsilon_\theta$对$z_t$学习$\mu_\theta(X_t, t) = \frac{1}{\sqrt{\bar\alpha_t}}(X_t -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(X_t,t))$  
最后$L_{T-1} 简化 L_{simple}(\theta):= \mathbb E_t, x_0, \epsilon[||z_t - \epsilon_\theta(X_t, t)||^2]$ 即对q扩散过程的噪声$z_t$对网络$\epsilon_\theta$输入(Xt, t)进行训练，此定义就是loss函数，能够最小化逆扩散过程$p_\theta$的$X_0$似然概率分布$p_\theta(X_0)$

