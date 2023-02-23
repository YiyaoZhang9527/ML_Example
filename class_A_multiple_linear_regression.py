import numpy as np
import matplotlib.pyplot as plt

'''
函数名称：多元线性回归
求解方法：多元线性回归函数如下：
符号注释：
Y : 理论因变量 
W : 各个自变量的权重参数
X : 是各个自变量

Y = W0+W1X1+W2X2+.....WnXn.
类似于：Y = W0X0+W1X1+W2X2+.....WnXn：其中X0=1(因为原始的函数W0的位置没有X的值，所以加了1，保持相乘后原值不变);
这样就相当于[W0,W1,W2...]的矩阵乘以[X0,X1,X2...]的转置矩阵.
x矩阵要添加一列1（因为X0==1。

因为，每一个样本点都会产生一个这样的矩阵方程，
这些方程中有W的参数矩阵是固定的，其他的变量都是样本点带进来的.
所以，我们要解出这个回归方程，其实就是在求解W的值。

直接求W这个一维矩阵是求不出来的，所以我们借助于最小二乘法。
最小二乘法：就是回归线的真实的y与理论Y之间的差值的平方，为最小。

符号注释：
y : 真实的y 
Y : 理论的Y
最小二乘法的图像是：y轴是Cost(Cost是花费的意思)值，就是(y-Y)^2.x轴就是W。
因为Y与y的差值最小只有一个未知数，就是W矩阵。所以在最小二乘法的公式中对W求导，在导数为0的时候，就是Cost最小，W最合理的地方。

公式推到如下：
对最小二乘法公式：（y-Y)^2的w求导，把 Y=wx带入，其中w,x为矩阵
=> (y-wx).T*(y-wx),求w导数
=> 2*(y-xw)*(y.T对w求偏导)-2*(y-xw)*[(xw).T对w求偏导].
=> 因为y是真实数没有w，所以y对W求偏导为0.所以上式退出：-2x.T(y-xw) = 0
=> x.T*x*w=x.T*y 
=> w = (x.T*x)^(-1)*x.T*y.其中矩阵的(-1)是逆矩阵的求法,
   样本矩阵的行数必须大于样本矩阵的列数，否则无法求逆矩阵
'''


# ：TODO 这是一个多元线性回归的函数
def multiple_linear_regression(Case_matrix, Result_matrix):
    '''
    函数名 : 多元线性回归(非梯度下降)
    :param Case_matrix: -> 自变量矩阵
    :param Result_matrix: -> 因变量矩阵
    :return: -> W : 参数矩阵
    '''
    x = np.mat(Case_matrix)
    y = np.mat(Result_matrix)
    # 判断x的行数要等于y的列数
    if (x.shape[0] == y.shape[1] and x.shape[0] >= x.shape[1] + 1):
        # x第一列添加1
        # q = np.row_stack((i, d)) 往i里面添加d,添加在后面
        # np.insert(x, 0, [1, 1], axis=1) Yes
        ones = np.ones((x.shape[0]))
        x = np.insert(x, 0, ones, axis=1)
        w = (x.T * x).I * (x.T) * (y.T)
        # print("W:"+str(w))
        return w
    else:
        print("input is error!")


if __name__ == '__main__':
    x = [[852, 2, 1, 36], [1534, 3, 2, 30], [1416, 3, 2, 40], [2104, 5, 1, 45], [1670, 4, 2, 38]]
    y = [178, 315, 232, 460, 380]
    print(multiple_linear_regression(x, y))
