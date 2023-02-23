# -*- encoding: utf-8 -*-
'''
@File    :   退火算法.py
@Time    :   2020/09/06 16:42:14
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib
import numpy as np

def f():  # 待优化最小函数
    for x in np.random.randn(1000):
        yield x

def PDE(DE, t, k=1):
    return np.exp((DE)/(k*t))

def DE_function(new, old):
    return new - old

def jump(DE, T, k=1):
    return PDE(DE, T, k) > np.random.rand() and 0 or 1

def simulated_annealing(func,parameter={"T": 1, "T_min": 0, "r": 0.0001, "expr": 0, "jump_max": np.inf}):
    path, funcpath = [], []
    T = parameter["T"]  # 系统温度，初时应在高温
    T_min = parameter["T_min"]  # 最小温度值
    r = parameter["r"]  # 降温速率
    counter = 0
    expr = parameter["expr"]  # 假设初解
    jump_max = parameter["jump_max"]  # 最大冷却值
    jump_counter = 0
    while T > T_min:
        counter += 1
        new_expr = func.__next__()  # 新解
        funcpath.append(new_expr)
        DE = new_expr - expr
        if DE <= 0:
            expr = new_expr
            jump_counter = 0
        elif DE > 0:
            expr = expr
            if jump(DE, T):
                T *= r
                jump_counter += 1
                if jump_counter > jump_max:
                    print("最大回炉冷却次数:", jump_counter)
                    return expr, path, funcpath
        path.append(expr)
        print("{}{}{}{}{}{}{}{}".format('系统温度:', T,
                                        ' 新状态:', expr, ' 迭代轮次:', counter, ' DE:', DE))

    return expr, path, funcpath

if __name__ == "__main__":
    expr, path, funcpath = simulated_annealing(f(),
    parameter={"T": 1, "T_min": 0, "r": 0.11, "expr": 0, "jump_max": 1000})
    print(expr)
