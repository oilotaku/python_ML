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
x =[x1,x2,x3,x4,x5]
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

def cost(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, Y):
    sum = 0
    m = len(x1)
    for i in range(0, m):
        sum += error(x1[i], x2[i], x3[i], x4[i], x5[i], w1, w2, w3, w4, w5, Y[i])
    return (1 / (2*m)) * sum

def rmse_w1(b,c, x, y):# 公式: ∇MSE(theta) = 2 / m * X.T * (X * theta - y) <- * 代表 dot
    squared_err = (b + c * x - y) ** 2
    res = numpy.sqrt(numpy.mean(squared_err))
    return res


def gradient_w1(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, Y, alpha, step):
    m = len(x1)
    points = []

    for i in range(0, step):
        prediction = x1 * w1
        errors = prediction - Y

        gradient = (1/m) * alpha *errors[i]
        w1 = w1 - gradient
        points.append(cost(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, Y))
    return w1


s = gradient_w1(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, y1, 0.04, 900)

print("w1: " + str(s))

