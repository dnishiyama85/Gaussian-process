"""
多変量ガウス分布からサンプリングして結果をプロット
"""


import numpy as np
import matplotlib.pyplot as plt


def _make_diff(xis, xjs):
    return np.array([(xi - xj) for xi in xis for xj in xjs]).reshape(len(xis),
                                                                     len(xjs))


def gauss_kernel(xis, xjs, theta1=1.0, theta2=1.0):
    diff = _make_diff(xis, xjs)
    return np.exp(-theta1 * diff ** 2 / theta2)


def exp_kernel(xis, xjs, theta1=1.0, theta2=1.0):
    diff = _make_diff(xis, xjs)
    return np.exp(-theta1 * np.abs(diff) / theta2)


def periodic_kernel(xis, xjs, theta1=1.0, theta2=1.0):
    diff = _make_diff(xis, xjs)
    return np.exp(theta1 * np.cos(diff) / theta2)


def sample(kernel, start=-5, stop=5, n=100, num_sample=3, random_x=False):
    if random_x:
        scale = stop - start
        xs = np.random.random(n)
        xs = xs * scale + start
    else:
        xs = np.linspace(-5, 5, n)
    mu = np.zeros_like(xs)
    k = kernel(xs, xs)
    return xs, np.random.multivariate_normal(mu, k, num_sample)


def plot(kernel):
    xs, yss = sample(kernel=kernel, num_sample=3)
    for ys in yss:
        plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    plot(kernel=periodic_kernel)
