from numpy import arange, sqrt
from numpy.polynomial import legendre
from numpy.random import uniform, randn

Q_f = arange(1, 100)
N = arange(20, 120, 5)
sigma_squared = arange(0, 2, 0.05)


def f(x, c):
    return legendre.legval(x=x, c=c)


def run_experiments():
    for num in N:
        for dim in Q_f:
            for sig in sigma_squared:
                # generate x values
                x = uniform(-1, 1, (num, dim))
                print(x)

                # a_q are the coefficients for the Legendre target function,
                # and are pulled from a standard normal distribution
                a_q = randn(dim)

                # generate y values
                y = []
                for x_i in x:
                    y.append(f(x_i[0], a_q) + sqrt(sig) * randn())
