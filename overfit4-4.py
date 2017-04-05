from numpy import arange, ceil, sqrt
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
    for num in N:
        for dim in Q_f:
            for sig in sigma_squared:
                g2_total = g10_total = 0
                for count in range(30):
                    # generate x values
                    k = int(ceil(num * 0.2))
                    x = uniform(-1, 1, (num))

                    # a_q are the coefficients for the Legendre target function,
                    # and are pulled from a standard normal distribution
                    a_q = randn(dim)

                    # generate y values
                    y = []
                    for x_i in x:
                        y.append(f(x_i, a_q) + sqrt(sig) * randn())

                    train_x = x[0:(num - k)]
                    test_x = x[(num - k):]
                    train_y = y[0:(num - k)]
                    test_y = y[(num - k):]

                    # g2_coeff and g10_coeff are coefficient matrices for a polynomial fit
                    g2_coeff = g_2(train_x, train_y)
                    g10_coeff = g_10(train_x, train_y)

                    g2 = 0
                    for i in range(len(test_x)):
                        g2 += (poly.polyval(test_x[i], g2_coeff) - test_y[i]) ** 2
                    g2_total += g2 / len(test_x)

                    g10 = 0
                    for i in range(len(test_x)):
                        g10 += (poly.polyval(test_x[i], g10_coeff) - test_y[i]) ** 2
                    g10_total += g10 / len(test_x)

                H2 = g2_total / 30
                H10 = g10_total / 30
                if(H10-H2 < lo):
                    lo = H10 - H2
                    print("for num: {1}, Q_f: {0}, sigma: {2:1.2f} -> Lo: H10-H2: {3}".format(dim, num, sig, H10 - H2))
                elif H10 - H2 > hi:
                    hi = H10 - H2
                    print("for num: {1}, Q_f: {0}, sigma: {2:1.2f} -> Hi: H10-H2: {3}".format(dim, num, sig, H10 - H2))
run_experiments()
