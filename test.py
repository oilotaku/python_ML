import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
df = pd.read_csv(r"./803_LRdata.csv")

x1 = np.array(df.x1.iloc[0:1000])
x2 = np.array(df.x2.iloc[0:1000])
x3 = np.array(df.x3.iloc[0:1000])
x4 = np.array(df.x4.iloc[0:1000])
x5 = np.array(df.x5.iloc[0:1000])
y1 = np.array(df.y1.iloc[0:900])
y2 = np.array(df.y2.iloc[0:900])
oy1 = np.array(df.y1.iloc[900:1000])
oy2 = np.array(df.y2.iloc[900:1000])
x1_5=np.array(df.iloc[0:1000,0:5])

def martix_f1(A,B):

    A_inv = np.linalg.inv(A)#Inverse matrix
    ans = A_inv.dot(B)

    return ans
rrange = 5;
mse = 0
ans1 = np.full((int(900/rrange), rrange), 0,dtype=float)
ans2 = np.full((int(900/rrange), rrange), 0,dtype=float)
for i in range(0, 900, rrange):
    ans1[int(i/rrange)] = martix_f1(x1_5[i:i+rrange], y1[i:i+rrange])
for i in range(0, 900, rrange):
    ans2[int(i/rrange)] = martix_f1(x1_5[i:i+rrange], y2[i:i+rrange])
for i in range(0, 180):
    for j in range(i*rrange+0, i*rrange+rrange):
        mse += (np.sum(ans1[i]*x1_5[j])-y1[j])**2# sigma(0,900)(t_y - r_y)^2
mse/900
print(mse)
print(ans1)
gd = np.array([1, 1, 1, 1, 1])
lr = 0.001
round_t = 0
mse1 = 1
'''while mse1>0:
    round_t+=1
    for i in range(0, 180):
        for j in range(i*rrange+0, i*rrange+rrange):
            y1_lr = np.longdouble(np.round(np.sum(gd * x1_5[j]), 4))
            y1_lrgd = np.longdouble(np.round((gd * x1_5[j]), 4))
            if (y1_lr < y1[j]):
                gd = gd + (y1_lrgd * lr)
            elif (y1_lr > y1[j]):
                gd = gd - (y1_lrgd * lr)
    for i in range(0, 180):
        for j in range(i * rrange + 0, i * rrange + rrange):
            mse1 = (np.sum(gd * x1_5[j]) - y1[j]) ** 2  # sigma(0,900)(t_y - r_y)^2
    print("mse1 :",mse1,"round_t ",round_t )
    time.sleep(1)
print(ans2[0])'''
lr = LinearRegression()
lr.fit(x1_5[0:900], y2)
print('Intercept:')
print(lr.intercept_)
print('\n')
print('Coefficient:')
print(lr.coef_)

y_pred = lr.predict(x1_5[0:900])
mse_validation = mean_squared_error(y2, y_pred)
print('MSE:')
print(mse_validation)
np.set_printoptions(precision = 2)
Y_o=lr.predict(x1_5[900:1000])
for i in range(0,2):
    plt.scatter(x1[i], Y_o[i], color='black')
    plt.scatter(x2[i], Y_o[i], color='blue')
    plt.scatter(x3[i], Y_o[i], color='red')
    plt.scatter(x4[i], Y_o[i], color='green')
    plt.scatter(x5[i], Y_o[i], color='yellow')
xp = np.linspace(1 ,40,100)
plt.plot(xp, xp*30, color='red')
plt.show()


'''
X2 = sm.add_constant(x1_5[0:900])
est = sm.OLS(y2, X2)
est2 = est.fit()
print(est2.summary())
def martix_function(x1,y,z,a,b,oup):
    D = np.linalg.det(np.vstack((x1, y, z, a, b)))
    Dx = np.linalg.det(np.vstack((oup, y, z, a, b)))
    Dy = np.linalg.det(np.vstack((x1, oup, z, a, b)))
    Dz = np.linalg.det(np.vstack((x1, y, oup, a, b)))
    Da = np.linalg.det(np.vstack((x1, y, z, oup, b)))
    Db = np.linalg.det(np.vstack((x1, y, z, a, oup)))
    inxd = np.array([Dx, Dy, Dz, Da, Db])
    inx = inxd / D
    return inx
ans1 = {}
ans2 = {}
for i in range(0,900,5):
    ans1[i/5] = martix_function(x1[i:i+5],x2[i:i+5],x3[i:i+5],x4[i:i+5],x5[i:i+5],y1[i:i+5])
for i in range(0,900,5):
    ans2[i/5] = martix_function(x1[i:i+5],x2[i:i+5],x3[i:i+5],x4[i:i+5],x5[i:i+5],y2[i:i+5])
'''