
## Answer 1

$$E_D (\mathbf{w}) = \frac{1}{2}\sum_{n=1}^Nr_n(t_n-\mathbf{w^T}\phi(\mathbf{x}_n))^2.$$

对误差函数$E_D (\mathbf{w})$关于$\mathbf{w}$求导，得到：

$$\frac{\partial E_D (\mathbf{w})}{\partial \mathbf{w}} = \sum_{n=1}^Nr_n(t_n-\mathbf{w^T}\phi(\mathbf{x}_n))(-\phi(\mathbf{x}_n))$$令导数为零，得到：

$$\sum_{n=1}^Nr_n(t_n-\mathbf{w^T}\phi(\mathbf{x}_n))(-\phi(\mathbf{x}_n)) = 0$$

$$\sum_{n=1}^Nr_n(t_n\phi(\mathbf{x}_n) - \mathbf{w^T}\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)) = 0$$

$$\sum_{n=1}^Nr_n\mathbf{w^T}\phi(\mathbf{x}_n)\phi(\mathbf{x}_n) = \sum_{n=1}^Nr_nt_n\phi(\mathbf{x}_n)$$

写成矩阵形式：

$$\mathbf{w^T}\left(\sum_{n=1}^Nr_n\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^T\right) = \sum_{n=1}^Nr_nt_n\phi(\mathbf{x}_n)$$

定义矩阵$\mathbf{A}$和向量$\mathbf{b}$如下：

$$\mathbf{A} = \sum_{n=1}^Nr_n\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^T$$

$$\mathbf{b} = \sum_{n=1}^Nr_nt_n\phi(\mathbf{x}_n)$$

解得：

$$\mathbf{w}^* = \mathbf{A}^{-1}\mathbf{b}$$


(i) 对于数据独立的噪声方差：将$r_n$看作是数据点$t_n$的权重，可以将$r_n$视为对应数据点的噪声方差的倒数。较大的$r_n$对应较小的噪声方差，反之亦然。

(ii) 对于重复的数据点：当存在重复的数据点时，将其权重$r_n$设置为重复出现的次数。这样，重复的数据点会在误差函数中具有更大的贡献，从而更强烈地影响参数$\mathbf{w}$的估计。

---

## **Answer 2**

根据贝叶斯定理，后验分布为：

$$p(\mathbf{w},\beta|\mathbf{t}) \propto p(\mathbf{t}|\mathbf{X},{\rm w},\beta) \cdot p(\mathbf{w},\beta)$$

代入最大似然与先验分布表达式如下：

$$p(\mathbf{t}|\mathbf{X},{\rm w},\beta) \cdot p(\mathbf{w},\beta) = \prod_{n=1}^{N}\mathcal{N}(t_n|{\rm w}^{\rm T}\phi({\rm x}_n),\beta^{-1}) \cdot \mathcal{N}(\mathbf{w|m}_0, \beta^{-1}\mathbf{S}_0) \cdot {\rm Gam}(\beta|a_0,b_0)$$

**对于$\mathbf{w}$的后验分布：**

忽略与$\mathbf{w}$无关的项（此时将 $Gam$ 项看作常数）：

$$p(\mathbf{w}|\mathbf{t}) \propto \prod_{n=1}^{N}\mathcal{N}(t_n|{\rm w}^{\rm T}\phi({\rm x}_n),\beta^{-1}) \cdot \mathcal{N}(\mathbf{w|m}_0, \beta^{-1}\mathbf{S}_0)$$

$$\ln p(\mathbf{w}|\mathbf{t}) \propto \sum_{n=1}^{N}\ln\mathcal{N}(t_n|{\rm w}^{\rm T}\phi({\rm x}_n),\beta^{-1}) + \ln\mathcal{N}(\mathbf{w|m}_0, \beta^{-1}\mathbf{S}_0)$$

将高斯分布的表达式代入上式中，并忽略与$\mathbf{w}$无关的项，得到：

$$\ln p(\mathbf{w}|\mathbf{t}) \propto -\frac{\beta}{2}\sum_{n=1}^{N}(t_n - {\rm w}^{\rm T}\phi({\rm x}_n))^2 - \frac{\beta}{2}(\mathbf{w-m}_0)^{\rm T}\mathbf{S}_0^{-1}(\mathbf{w-m}_0)$$

$$\ln p(\mathbf{w}|\mathbf{t}) \propto -\frac{\beta}{2}\left[\sum_{n=1}^{N}(t_n - {\rm w}^{\rm T}\phi({\rm x}_n))^2 + (\mathbf{w-m}_0)^{\rm T}\mathbf{S}_0^{-1}(\mathbf{w-m}_0)\right]$$
与高斯分布的指数项进行比较，得到后验分布的参数：

$$\mathbf{S}_N^{-1} = \beta\sum_{n=1}^{N}\phi({\rm x}_n)\phi({\rm x}_n)^{\rm T} + \mathbf{S}_0^{-1}$$

$$\mathbf{m}_N = \mathbf{S}_N\left[\beta\sum_{n=1}^{N}t_n\phi({\rm x}_n) + \mathbf{S}_0^{-1}\mathbf{m}_0\right]$$

**对于$\beta$的后验分布：

展开并忽略与$\beta$无关的项：

$$p(\beta|\mathbf{t}) \propto \prod_{n=1}^{N}\mathcal{N}(t_n|{\rm w}^{\rm T}\phi({\rm x}_n),\beta^{-1}) \cdot {\rm Gam}(\beta|a_0,b_0)$$

$$\ln p(\beta|\mathbf{t}) \propto \sum_{n=1}^{N}\ln\mathcal{N}(t_n|{\rm w}^{\rm T}\phi({\rm x}_n),\beta^{-1}) + \ln{\rm Gam}(\beta|a_0,b_0)$$

将高斯分布的表达式代入上式中，并忽略与$\beta$无关的项，得到：

$$\ln p(\beta|\mathbf{t}) \propto -\frac{\beta}{2}\sum_{n=1}^{N}(t_n - {\rm w}^{\rm T}\phi({\rm x}_n))^2 + \ln{\rm Gam}(\beta|a_0,b_0)$$

与伽玛分布的指数项进行比较，可以得到后验分布参数：

$$a_N = a_0 + \frac{N}{2}$$

$$b_N = b_0 + \frac{1}{2}\sum_{n=1}^{N}(t_n - {\rm w}^{\rm T}\phi({\rm x}_n))^2$$

**综上所述：**

后验分布的形式为：

$$p(\mathbf{w},\beta|\mathbf{t}) = \mathcal{N}(\mathbf{w|m}_N, \beta^{-1}\mathbf{S}_N) \cdot {\rm Gam}(\beta|a_N,b_N)$$

其中后验参数的表达式为：

$$\mathbf{S}_N^{-1} = \beta\sum_{n=1}^{N}\phi({\rm x}_n)\phi({\rm x}_n)^{\rm T} + \mathbf{S}_0^{-1}$$

$$\mathbf{m}_N = \mathbf{S}_N\left[\beta\sum_{n=1}^{N}t_n\phi({\rm x}_n) + \mathbf{S}_0^{-1}\mathbf{m}_0\right]$$

$$a_N = a_0 + \frac{N}{2}$$

$$b_N = b_0 + \frac{1}{2}\sum_{n=1}^{N}(t_n - {\rm w}^{\rm T}\phi({\rm x}_n))^2$$


---

## **Answer 3**

**（1）证明：**

在贝叶斯线性回归模型中对$\mathbf{w}$的积分给出的结果，首先考虑贝叶斯线性回归模型中$\mathbf{w}$的指数项的积分：

$$\int \exp\{-E(\mathbf{w})\} {\rm d}\mathbf{w}$$

其中，$E(\mathbf{w})$是定义在$\mathbf{w}$上的能量函数。可将$E(\mathbf{w})$写成二次型的形式：

$$E(\mathbf{w}) = \frac{1}{2}(\mathbf{w}-\mathbf{m}_N)^T\mathbf{A}(\mathbf{w}-\mathbf{m}_N)$$

$\mathbf{m}_N$是均值向量，$\mathbf{A}$是协方差矩阵。

通过比较二次型的形式和高斯分布的指数项，可以看出$\mathbf{A}$在这里作为协方差矩阵。

将能量函数$E(\mathbf{w})$代入积分中，得到：

$$\int \exp\{-\frac{1}{2}(\mathbf{w}-\mathbf{m}_N)^T\mathbf{A}(\mathbf{w}-\mathbf{m}_N)\} {\rm d}\mathbf{w}$$

我们可以通过引入一个标准化常数来将指数项转化为一个多元高斯分布的归一化常数。这个标准化常数与$\mathbf{A}$的行列式成比例。因此，将积分表示为：

$$\int \exp\{-\frac{1}{2}(\mathbf{w}-\mathbf{m}_N)^T\mathbf{A}(\mathbf{w}-\mathbf{m}_N)\} {\rm d}\mathbf{w} = \exp\{-E(\mathbf{m}_N)\}|\mathbf{A}|^{-1/2}(2\pi)^{M/2}$$

证毕

**（2）证明：**

$$\ln p(\mathbf{t}|\alpha,\beta) = \frac{M}{2}\ln\alpha + \frac{N}{2}\ln\beta - E(\mathbf{m}_N) - \frac{1}{2}\ln|\mathbf{A}| - \frac{N}{2}\ln(2\pi)$$

将对数边缘似然写为：

$$\ln p(\mathbf{t}|\alpha,\beta) = \ln \int p(\mathbf{t}|\mathbf{w},\beta)p(\mathbf{w}|\alpha) {\rm d}\mathbf{w}$$

已知$\mathbf{w}$的后验分布，其形式为高斯分布：

$$p(\mathbf{w}|\mathbf{t},\alpha,\beta) = \mathcal{N}(\mathbf{w}|\mathbf{m}_N,\mathbf{A}^{-1})$$

将上式代入对数边缘似然的表达式中，得到：

$$\ln p(\mathbf{t}|\alpha,\beta) = \ln \int p(\mathbf{t}|\mathbf{w},\beta)\mathcal{N}(\mathbf{w}|\mathbf{m}_N,\mathbf{A}^{-1}) {\rm d}\mathbf{w}$$

将似然函数和后验分布的指数项合并，并利用高斯分布的性质化简，得到最终的表达式：

$$\ln p(\mathbf{t}|\alpha,\beta) = \frac{M}{2}\ln\alpha + \frac{N}{2}\ln\beta - E(\mathbf{m}_N) - \frac{1}{2}\ln|\mathbf{A}| - \frac{N}{2}\ln(2\pi)$$

其中，$M$是$\mathbf{w}$的维度，$N$是训练数据的数量。证毕

---

## Answer 4
$$F(a)=\frac{1}{2}\sum_{i}(Y_i-aX_i)^2$$

对$F(a)$求导，得到：

$$\frac{dF(a)}{da} = -\sum_{i}X_i(Y_i-aX_i)$$

将导数等于0，得到：

$$-\sum_{i}X_i(Y_i-aX_i) = 0$$

$$\sum_{i}X_iY_i - a\sum_{i}X_i^2 = 0$$

$$a\sum_{i}X_i^2 = \sum_{i}X_iY_i$$

解出$a$的表达式：

$$a = \frac{\sum_{i}X_iY_i}{\sum_{i}X_i^2}$$


---

## Answer 5

给定独立从参数为$\theta$的泊松分布中抽取的数据点$y_1, \dots ,y_n$，将数据的对数似然函数写成关于$\theta$的函数。对于单个观测$y$，其概率可以表示为：

$$p(y|\theta)=\frac{\theta^{y}e^{-\theta}}{y!}, {\rm for}\;y = 0, 1, 2,\dots$$

而对于给定的数据点$y_1, \dots ,y_n$，它们是独立的，因此它们的联合概率可以写成各个数据点概率的乘积形式：

$$p(y_1, \dots ,y_n|\theta) = \prod_{i=1}^{n} p(y_i|\theta)$$

因此，对数似然函数可以写成：

$$\ln p(y_1, \dots ,y_n|\theta) = \ln\left(\prod_{i=1}^{n} p(y_i|\theta)\right)$$

利用对数的性质，将乘积转化为求和：

$$\ln p(y_1, \dots ,y_n|\theta) = \sum_{i=1}^{n} \ln p(y_i|\theta)$$

将$p(y|\theta)$的定义代入，得到：

$$\ln p(y_1, \dots ,y_n|\theta) = \sum_{i=1}^{n} \ln \left(\frac{\theta^{y_i}e^{-\theta}}{y_i!}\right)$$

进一步展开和整理，可以得到：

$$\ln p(y_1, \dots ,y_n|\theta) = \sum_{i=1}^{n} \left(y_i\ln\theta - \theta - \ln(y_i!)\right)$$

---

## Answer 6

$$L(\lambda) = \prod_{i=1}^{n} f_X(X_i)$$

取对数似然函数，可以将乘积转化为求和：

$$\ln L(\lambda) = \sum_{i=1}^{n} \ln f_X(X_i)$$

代入$Gamma(\alpha,\lambda)$分布的概率密度函数，得到：

$$\ln L(\lambda) = \sum_{i=1}^{n} \ln \left(\frac{1}{\Gamma(\alpha)}\lambda^{\alpha}X_i^{\alpha-1}e^{-\lambda X_i}\right)$$

进一步展开和整理，可以得到：

$$\ln L(\lambda) = -n\ln \Gamma(\alpha) + n\alpha\ln \lambda + (\alpha-1)\sum_{i=1}^{n} \ln X_i - \lambda\sum_{i=1}^{n} X_i$$

对$\ln L(\lambda)$关于$\lambda$求导，得到：

$$\frac{d\ln L(\lambda)}{d\lambda} = \frac{n\alpha}{\lambda} - \sum_{i=1}^{n} X_i$$

$$\frac{n\alpha}{\lambda} - \sum_{i=1}^{n} X_i = 0$$

$$\frac{n\alpha}{\lambda} = \sum_{i=1}^{n} X_i$$

解出$\lambda$的表达式：
$$\lambda = \frac{n\alpha}{\sum_{i=1}^{n} X_i}$$
