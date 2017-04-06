import cProfile

import numpy as np
from numpy.polynomial import legendre
from numpy.polynomial.polynomial import polyfit, polyval
from numpy.random import uniform, randn

Q_f = np.arange(1, 100, 10)
N = np.arange(20, 120, 5)
sigma_squared = np.arange(0, 2, 0.5)


def f(x, c):
    return sum(legendre.legval(x=x, c=c))


def g_2(x, y):
    return polyfit(x[:, 1], y, 2)


def g_10(x, y):
    return polyfit(x[:, 1], y, 10)


def run_experiments():
    lo = hi = 1
    experiment_size = 100

    for num in N:                               # 100
        y = np.zeros(num)
        print("num: {0}".format(num))
        k = int(np.ceil(num * 0.2))
        diff = num - k

        for dim in Q_f:  # 20
            print("\tQ_f dim: {0}".format(dim))
            # a_q are the coefficients for the Legendre target function,
            # and are pulled from a standard normal distribution
            a_q = randn(dim)

            for sig in sigma_squared:  # 40
                g2_sq_err = g10_sq_err = 0  # = 80,000 runs, 100 experiments each,
                sq = np.sqrt(sig)

                for count in range(experiment_size):  # = 8,000,000 runs
                    # generate x values
                    x = uniform(-1, 1, (num,1))
                    ones = np.ones((num, 1))
                    x = np.concatenate((ones, x), axis=1)

                    # generate y values
                    for i in range(len(x)):
                        y[i] = f(x[i], a_q) + sq * randn()

                    # break into test and training sets
                    train_x, test_x = x[:diff], x[diff:]
                    train_y, test_y = y[:diff], y[diff:]

                    #  g2_coeff and g10_coeff are coefficient matrices for a polynomial fit
                    g2_coeff = g_2(train_x, train_y)
                    g10_coeff = g_10(train_x, train_y)

                    # calculate squared error for g2
                    g2_sq_err += sum((polyval(test_x[i][1], g2_coeff) - test_y[i]) ** 2 for i in range(k)) / k

                    # calculate squared error for g10
                    g10_sq_err += sum((polyval(test_x[i][1], g10_coeff) - test_y[i]) ** 2 for i in range(k)) / k

                H2 = g2_sq_err / experiment_size
                H10 = g10_sq_err / experiment_size
                overfit = H10 - H2

                if overfit < lo:
                    lo = overfit
                    print("\t\tfor sigma: {0:1.2f} -> Lo: H10 - H2: {1:.4f}".format(sig, overfit))
                elif overfit > hi:
                    hi = overfit
                    print("\t\tfor sigma: {0:1.2f} -> Hi: H10 - H2: {1:.4f}".format(sig, overfit))

cProfile.run('run_experiments()')
