import pandas as pd
import numpy as np

# 导入数据
train = pd.read_csv('803_LRdata.csv')

test = pd.read_csv('803_LRdata.csv')
submit = pd.read_csv('803_LRdata.csv')

# 初始设置
beta = [1, 1]
alpha = 0.01
tol_L = 0.1

max_x1 = max(train['x1'])
x1 = train['x1'] / max_x1

max_x2 = max(train['x2'])
x2 = train['x2'] / max_x2

max_x3 = max(train['x3'])
x3 = train['x3'] / max_x3

max_x4 = max(train['x4'])
x4 = train['x4'] / max_x4

max_x5 = max(train['x5'])
x5 = train['x5'] / max_x5

y1 = train['y1']

# 计算梯度的函數
def compute_grad(beta, x, y):
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)

# 更新beta的函数
def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# 计算RMSE的函數
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

# 第一次计算
grad = compute_grad(beta, x1, y1)
loss = rmse(beta, x1, y1)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x1, y1)

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, x1, y1)
    loss = loss_new
    loss_new = rmse(beta, x1, y1)
    i += 1


print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
print('Coef: %s \nw1 %s'%(beta[1], beta[0]))
