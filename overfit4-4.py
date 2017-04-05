from numpy import arange, sqrt
from numpy.polynomial import legendre, polynomial as poly
from numpy.random import uniform, randn

Q_f = arange(1, 100)
N = arange(20, 120, 5)
sigma_squared = arange(0, 2, 0.05)


def f(x, c):
    return legendre.legval(x=x, c=c)


def g_2(x, y):
    return poly.polyfit(x, y, 2)


def g_10(x, y):
    return poly.polyfit(x, y, 10)


def run_experiments():
    for num in N:
        for dim in Q_f:
            for sig in sigma_squared:
                # generate x values
                k = num * 0.2
                x = uniform(-1, 1, (num, 1))

                # a_q are the coefficients for the Legendre target function,
                # and are pulled from a standard normal distribution
                a_q = randn(dim)

                # generate y values
                y = []
                for x_i in x:
                    y.append(f(x_i[0], a_q) + sqrt(sig) * randn())

                train_x = x[0:num - k], test_x = x[num - k:]
                train_y = y[0:num - k], test_y = y[num - k:]

                # g2_coeff and g10_coeff are coefficient matrices for a polynomial fit
                g2_coeff = g_2(train_x, train_y)
                g10_coeff = g_10(train_x, train_y)

                g2_sum = 0
                for i in len(test_x):
                    g2_sum += (poly.polyval(test_x[i], g2_coeff) - test_y[i]) ** 2

                g10_sum = 0
                for i in len(test_x):
                    g10_sum += (poly.polyval(test_x[i], g10_coeff) - test_y[i]) ** 2
