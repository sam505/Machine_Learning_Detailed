from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
y = np.array([5, 4, 6, 5, 8, 7], dtype=np.float64)


def slope_and_intercept_line_best_fit(x, y):
    gradient = ((mean(x) * mean(y)) - (mean(x * y))) / ((mean(x)) ** 2 - mean(x ** 2))
    y_intercept = mean(y) - (gradient * mean(x))

    return gradient, y_intercept


m, b = slope_and_intercept_line_best_fit(x, y)
print(m)

print(b)

regression_line = [m * i + b for i in x]
print(regression_line)

plt.scatter(x=x, y=y)
plt.plot(x, regression_line)
plt.show()


def squared_error(y_original, y_line):
    return sum((y_line - y_original) ** 2)


def coefficient_of_determination(y_original, y_line):
    y_mean_line = [mean(y_original) for _ in y_original]
    squared_error_regression = squared_error(y_original, y_line)
    squared_error_y_mean = squared_error(y_original, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)


r_squared = coefficient_of_determination(y, regression_line)
print(r_squared)