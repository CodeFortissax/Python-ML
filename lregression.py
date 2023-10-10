from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_dataset(40, 40, 2, correlation='pos')


def best_fit_slope_and_intercept(xs, ys):
    mx = mean(xs)
    my = mean(ys)
    mxy = mean(xs*ys)
    m = (((mx*my)-mxy) / (mx**2 - mean(xs**2)))
    b = my-(m*mx)
    return m, b


def determination_coefficient():
    sst = np.sum(np.power(regression_line - ys, 2))
    sse = np.sum(np.power(ys - np.mean(ys), 2))
    return 1 - sst / sse


m, b = best_fit_slope_and_intercept(xs, ys)
print(m)
print(b)

regression_line = [(m*x)+b for x in xs]


r_squared = determination_coefficient()
print(r_squared)


predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs,ys)
#plt.scatter(predict_x, predict_y, color ='g')
plt.plot(xs, regression_line)
plt.show()