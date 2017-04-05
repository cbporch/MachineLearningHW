from numpy import arange, ceil, sqrt, zeros
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
    lo = hi = 1

    for num in N:                               # 100
        y = zeros(num)
        print("num: {0}".format(num))
        k = int(ceil(num * 0.2))
        diff = num - k
        for dim in Q_f:                         # 20
            print("\tQ_f dim: {0}".format(dim))

            for sig in sigma_squared:           # 40
                g2_sq_err = g10_sq_err = 0      # = 80,000 runs, 100 experiments each,
                sq = sqrt(sig)

                for count in range(100):        # = 8,000,000 runs
                    # generate x values
                    x = uniform(-1, 1, num)

                    # a_q are the coefficients for the Legendre target function,
                    # and are pulled from a standard normal distribution
                    a_q = randn(dim)

                    # generate y values
                    for i in range(len(x)):
                        y[i] = f(x[i], a_q) + sq * randn()

                    # break into test and training sets
                    train_x, test_x = x[:diff], x[diff:]
                    train_y, test_y = y[:diff], y[diff:]

                    # g2_coeff and g10_coeff are coefficient matrices for a polynomial fit
                    g2_coeff = g_2(train_x, train_y)
                    g10_coeff = g_10(train_x, train_y)

                    g2 = 0
                    for i in range(len(test_x)):
                        g2 += (poly.polyval(test_x[i], g2_coeff) - test_y[i]) ** 2
                    g2_sq_err += g2 / len(test_x)

                    g10 = 0
                    for i in range(len(test_x)):
                        g10 += (poly.polyval(test_x[i], g10_coeff) - test_y[i]) ** 2
                    g10_sq_err += g10 / len(test_x)

                H2 = g2_sq_err / 100
                H10 = g10_sq_err / 100
                overfit = H10 - H2

                if overfit < lo:
                    lo = overfit
                    print("\t\tfor sigma: {0:1.2f} -> Lo: H10-H2: {1}".format(sig, overfit))
                elif overfit > hi:
                    hi = overfit
                    print("\t\tfor sigma: {0:1.2f} -> Hi: H10-H2: {1}".format(sig, overfit))
run_experiments()
