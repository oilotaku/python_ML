import numpy
import pandas
smp_data = pandas.read_csv('803_LRdata.csv')
x1 = numpy.array(smp_data.x1.iloc[0:1000])
x2 = numpy.array(smp_data.x2.iloc[0:1000])
x3 = numpy.array(smp_data.x3.iloc[0:1000])
x4 = numpy.array(smp_data.x4.iloc[0:1000])
x5 = numpy.array(smp_data.x5.iloc[0:1000])
y1 = numpy.array(smp_data.y1.iloc[0:1000])
y2 = numpy.array(smp_data.y2.iloc[0:1000])

w1 =1
w2 =1
w3 =1
w4 =1
w5 =1

weights = 1
def predict(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5):
    return w1 * x1 + w2 * x2 + w3* x3 + w4 * x4 + w5 * x5

def error(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, y1):
    return (predict(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5) - y1)**2

def rmse_w1(w, x, y):# 公式: ∇MSE(theta) = 2 / m * X.T * (X * theta - y) <- * 代表 dot
    squared_err = (w * x - y) ** 2
    res = numpy.sqrt(numpy.mean(squared_err))
    return res

def gradient_w1(x1, w1, Y, alpha, step):
    m = len(x1)
    a = 0
    loss = rmse_w1 (w1, x1[0], y1[0])
    loss_new = rmse_w1 (w1, x1[0], y1[0])

    for i in range(0, step):
        prediction = x1 * w1
        errors =  prediction - Y

        loss = loss_new
        loss_new = rmse_w1 (w1, x1[i+1], y1[i+1])
        a = numpy.abs(loss_new - loss)
        if (a <= alpha):
            break
        gradient = (2/m) * alpha *errors[i]
        w1 = w1 - gradient
    return w1



s = gradient_w1(x1,w1,y1, 0.1, 900)
s2 = gradient_w1(x2,w2,y1, 0.1, 900)
s3 = gradient_w1(x3,w3,y1, 0.1, 900)
s4 = gradient_w1(x4,w4,y1, 0.1, 900)
s5 = gradient_w1(x5,w5,y1, 0.05, 900)


print("w1: " + str(s))
print("w2: " + str(s2))
print("w3: " + str(s3))
print("w4: " + str(s4))
print("w5: " + str(s5))
print(predict(x1[0], x2[0], x3[0], x4[0], x5[0], s, s2, s3, s4, s5))
print(y1[0])

