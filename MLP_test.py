import numpy
import pandas

smp_data = pandas.read_csv('803_LRdata.csv')
x1 = numpy.array(smp_data.x1.iloc[0:900])
x2 = numpy.array(smp_data.x2.iloc[0:900])
x3 = numpy.array(smp_data.x3.iloc[0:900])
x4 = numpy.array(smp_data.x4.iloc[0:900])
x5 = numpy.array(smp_data.x5.iloc[0:900])
y1 = numpy.array(smp_data.y1.iloc[0:900])
y2 = numpy.array(smp_data.y2.iloc[0:900])

x1 = numpy.array(x1).reshape(900,1)
x2 = numpy.array(x2).reshape(900,1)
x3 = numpy.array(x3).reshape(900,1)
x4 = numpy.array(x4).reshape(900,1)
x5 = numpy.array(x5).reshape(900,1)


x = [ x1, x2, x3, x4, x5]
x = numpy.array(x).reshape(900,5)

w = numpy.random.randn( 5, 15)
y = numpy.random.randn( 15, 1)

w = x.dot(w)
y = w.dot(y) 

def predict(x,y):
    return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]+x[3]*y[3]+x[4]*y[4]


