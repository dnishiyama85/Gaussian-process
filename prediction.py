import numpy as np
import matplotlib.pyplot as plt

from sampling import sample, gauss_kernel


def get_train_data(size=10, start=-5, stop=5):
    xs, ys = sample(gauss_kernel, start=start, stop=stop, n=size,
                    num_sample=1, random_x=True)
    return xs, ys[0]


# 予測分布はガウス分布になるので、その平均と分散を返す
def predict_distribution(xs, train_xs, train_ys, theta1=1.0, theta2=1.0):
    k_ = gauss_kernel(train_xs, xs)
    k__ = gauss_kernel(xs, xs)
    K = gauss_kernel(train_xs, train_xs, theta1=theta1, theta2=theta2)
    K_inv = np.linalg.inv(K)
    means = k_.T.dot(K_inv.dot(train_ys))
    vars = k__ - k_.T.dot(K_inv).dot(k_)
    return means, vars


if __name__ == '__main__':
    start = -5
    stop = 5
    train_xs, train_ys = get_train_data(start=start, stop=stop)
    xs = np.linspace(start, stop, 100)
    means, vars = predict_distribution(xs, train_xs, train_ys,
                                       theta1=1.0, theta2=1.0)
    # 分散は対角成分のみを取り出して表示する
    vars = np.diag(vars)
    std = np.sqrt(vars)
    upper = means + std
    lower = means - std
    plt.scatter(train_xs, train_ys, c='r')
    plt.plot(xs, means)
    plt.fill_between(xs, upper, lower, facecolor='y', alpha=0.5)
    plt.show()

