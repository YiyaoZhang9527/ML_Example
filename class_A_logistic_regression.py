import numpy as np
import matplotlib.pyplot as plt

from class_filetools_module import mkdir
from os import getcwd

dirpath = getcwd() + '/plot_file/logistic_regression/'
mkdir(dirpath).__repr__()

# ：TODO 这是逻辑回归函数
'''参数列表处理'''


def loaddata(filename):
    file = open(filename)
    x = []
    y = []
    for line in file.readlines():
        line = line.strip().split()
        x.append([1, float(line[0]), float(line[1])])
        y.append(float(line[-1]))
    xmat = np.mat(x)
    ymat = np.mat(y).T
    file.close()
    return xmat, ymat


'''求出最优参数W'''


def w_calc(xmat, ymat, alpha=0.001, maxIter=10001):
    # w init
    w = np.mat(np.random.randn(3, 1))
    # w update
    for i in range(maxIter):
        H = 1 / (1 + np.exp(-xmat * w))
        dw = xmat.T * (H - ymat)  # dw:(3,1)
        w -= alpha * dw
    return w


if __name__ == '__main__':
    '''运行计算部分'''
    xmat, ymat = loaddata('ytb_lr.txt')
    w = w_calc(xmat, ymat)
    print('w', w)

    '''作图部分'''
    w0 = w[0, 0]
    w1 = w[1, 0]
    w2 = w[2, 0]
    plotx1 = np.arange(1, 7, 0.01)
    plotx2 = -w0 / w2 - w1 / w2 * plotx1
    plt.plot(plotx1, plotx2, c='r', label='decision boundary')
    plt.scatter(xmat[:, 1][ymat == 0].A,
                xmat[:, 2][ymat == 0].A,
                marker='^',
                s=150)
    plt.scatter(xmat[:, 1][ymat == 1].A, xmat[:, 2][ymat == 1].A, s=150)
    plt.grid()
    plt.legend()
    plt.savefig(dirpath + 'LogisticRegressionFromMofei.png')
    plt.show()
