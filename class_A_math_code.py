import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)  # 显示numpy数据结构的大文件
import operator


# 求素数方法
def primeNumber(n):
    n = 10000
    sum = 0
    prime = []
    for i in range(n + 1):
        prime.append(True)
    for i in range(2, n + 1):
        if prime[i]:
            print(i),
            j = i + i
            while j <= n:
                prime[j] = False
                j += i

# n是元素数，m是取值数
from functools import reduce


def Amn(Range1, NumberOfCombinations):  # 排列计算函数
    reducen = reduce(lambda x, y: x * y, (range(1, Range1 + 1)))  # 所有元素个数 （n!）的阶乘
    reducen_m = reduce(lambda z, y: z * y,
                       (range(1, (Range1 - NumberOfCombinations) + 1)))  # 所有元素个数的阶乘 减去 组合元素每个长度的阶乘（n!-m!）
    return reducen / reducen_m  # (n!)/(n!-m!) 表示共有多少种组合


def Cmn(Range1, NumberOfCombinations):  # 组合计算函数
    reducen = reduce(lambda x, y: x * y, (range(1, Range1 + 1)))  # 所有元素个数 （n!）的阶乘
    reducem_nm = reduce(lambda x, y: x * y, (range(1, NumberOfCombinations + 1))) * reduce(lambda z, y: z * y, (
        range(1, (Range1 - NumberOfCombinations) + 1)))  # m!*(n-m)!
    return reducen / reducem_nm  # n!/ m!*(n-m)!


# 复利计算
def myfv(rate, nper, pmt):
    '''
    复利计算
    :param rate:
    :param nper:
    :param pmt:
    :return:
    '''
    return np.fv(rate, nper, -pmt, -pmt)


dx05 = [myfv(0.05, x, 1) for x in range(0, 50)]
dx08 = [myfv(0.08, x, 1) for x in range(0, 50)]
dx10 = [myfv(0.10, x, 1) for x in range(0, 50)]
df = pd.DataFrame(columns=['dx05', 'dx08', 'dx10'])
df['dx05'] = dx05[:]
df['dx08'] = dx08
df['dx10'] = dx10
print(df.tail(1))
df.plot()

# 八个方向坐标计算
north = lambda parent_node: [parent_node[0], parent_node[-1] - 1]  #
northeast = lambda parent_node: [parent_node[0] + 1, parent_node[-1] - 1]  #
east = lambda parent_node: [parent_node[0] + 1, parent_node[-1]]  #
southeast = lambda parent_node: [parent_node[0] + 1, parent_node[-1] + 1]  #
south = lambda parent_node: [parent_node[0], parent_node[-1] + 1]  #
southwest = lambda parent_node: [parent_node[0], parent_node[-1]]  #
west = lambda parent_node: [parent_node[0], parent_node[-1]]  #
northwest = lambda parent_node: [parent_node[0], parent_node[-1]]  #

Compare_for_2_kind = lambda data1, data2: operator.eq(data1, data2)  # openator.eq比较两个任意对象

Compare_for_3_kind = lambda left, right: 'left<right' if (left > right) - (left < right) < 0 else (
    'left=right' if (left > right) - (left < right) == 0 else 'left>right')  # 三元比较


# 复利计算
def compoud_interest(p, i, n):
    p = float(p)  # 本金
    i = float(i)  # 利率
    n = int(n)  # 期数
    # 复利公式：s = p(1 + i)n
    return p * (1 + i) ** n


# 调包计算排列组合
from itertools import combinations, permutations

example_list = list(range(4))
per = list(permutations(example_list, 2))  # 组合计算
com = list(combinations(example_list, 2))  # 排列计算


# 排列组合函数

def Amn1(Range1, NumberOfCombinations, key):
    from scipy.special import comb, perm
    perm_value = 'perm', perm(Range1, NumberOfCombinations)
    comb_value = 'comb', comb(Range1, NumberOfCombinations)
    key_dict = {'key1': perm_value, 'key2': comb_value}
    return key_dict[str(key)]


def Amn2(Range1, NumberOfCombinations, key):  # 排列计算函数
    from functools import reduce
    reducen = reduce(lambda x, y: x * y, (range(1, Range1 + 1)))  # 所有元素个数 （n!）的阶乘
    reducen_m = reduce(lambda z, y: z * y,
                       (range(1, (Range1 - NumberOfCombinations) + 1)))  # 所有元素个数的阶乘 减去 组合元素每个长度的阶乘（n!-m!）
    value1 = reducen / reducen_m  # (n!)/(n!-m!) 表示共有多少种组合
    reducen = reduce(lambda x, y: x * y, (range(1, Range1 + 1)))  # 所有元素个数 （n!）的阶乘
    reducem_nm = reduce(lambda x, y: x * y, (range(1, NumberOfCombinations + 1))) * reduce(lambda z, y: z * y, (
        range(1, (Range1 - NumberOfCombinations) + 1)))  # m!*(n-m)!
    value2 = reducen / reducem_nm  # n!/ m!*(n-m)!
    key_dict = {'key1': value1, 'key2': value2}
    return key_dict[str(key)]


def Amn3(Range1, NumberOfCombinations):
    m = Range1 - NumberOfCombinations + 1
    p = 1
    for i in range(1, NumberOfCombinations):
        m *= (i + m)
        p *= (i + 1)
    return m / p


def Amn4(Range1, NumberOfCombinations):
    m = Range1 - NumberOfCombinations + 1
    p, mm = 1, 1
    for i in range(0, NumberOfCombinations):
        mm *= i + m
        p *= i + 1
    return mm / p


def element_per_com(range1, NumberOfCombinations, key):
    from itertools import combinations, permutations
    example_list = [i for i in range(range1)]
    return key == 1 and list(permutations(example_list, NumberOfCombinations)) or list(
        combinations(example_list, 2))  # 组合枚举计算


LGG = lambda X, n, a, c, m: X(n + 1) == (a * X(n) + c) % m


def XI(A, C, Xi, M):
    '''
    同余算法
    Xi = (Xi-1 * A + C ) mod M
其中A,C,M都是常数（一般会取质数）。当C=0时，叫做乘同余法。引出一个概
念叫seed，它会被作为X0被代入上式中，然后每次调用rand()函数都会用上一
次产生的随机值来生成新的随机值。可以看出实际上用rand()函数生成的是一个
递推的序列，一切值都来源于最初的 seed。所以当初始的seed取一样的时候，
得到的序列都相同。C语言里面有RAND_MAX这样一个宏，定义了rand()所能得到
的随机值的范围。在C里可以看到RAND_MAX被定义成0x7fff，也就是32767。
rand()函数里递推式中M的值就是32767。
    :param A:
    :param C:
    :param Xi:
    :param M:
    :return:
    '''
    for Xi_1 in range(Xi - 1 * 2, 0):
        Xi_C = abs(Xi) * A + C
        if C == 0:
            return Xi(A, Xi_C, Xi, M)


def P():
    '''
    素数算法
    :return:
    '''
    n = 1000
    sum = 0
    prime = []
    for i in range(n + 1):
        prime.append(True)
    for i in range(2, n + 1):
        if prime[i]:
            print(i),
            j = i + i
            while j <= n:
                prime[j] = False
                j += i


# 圆周率
def pi():
    '''
    圆周率
    :return:
    '''
    s = 0
    n = 1
    while n < 1e7:
        s += (-1) ** (n - 1) / (2 * n - 1)
        n += 1
    return (s * 4)


def multi(listdata=[0.2,0.8,0.8,0.8,0.8]):
    '''
    累乘函数
    :param listdata: 给一个可以list的数据结构向量
    :return:
    '''
    n = 1
    for i in listdata:
        n *= i
        print(n)
    return n

def factorial(n):
    '''
    阶乘函数
    :param n: 输入一个向量的总元素范围
    :return:
    '''
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def Inverse(number):
    '''
    构造一个单位矩阵
    :param number: 输入矩阵的总元素数，必须是可以构成正方形的矩阵
    :return:
    '''
    number1 = np.sqrt(number).astype(int)
    return np.array([i % (number1 + 1)== 0 and True or False for i in range(number)]).reshape(number1,number1).astype(int)
Inverse(100)

import math
import time
import scipy

math.sqrt(1000)  # 平方根
math.factorial(1000)  # 阶乘
pow = pow(2, 10)  # 阶乘
pi = math.pi  # 圆周率
math_pow = math.pow(2, 10)
scipy.pi  # 圆周率
scipy.rand(10)  # 随机10个数
math.e  # 自然常数
math.fabs(-1)  # 绝对值
math.exp(100.111)  # 指数函数
scipy.log2(2)  # log以2为底8的对数
scipy.log(10)  # 返回自然对数
a = 3
b = 4
print(math.hypot(a,b)) #求斜边
print(np.sqrt(((-3)**2)+((-4)**2))) #同上，求斜边
#scipy.log(125, 5)  # 返回以125为底5的对数     通过log(x[, base])来设置底数，如 log(x, 10) 表示以10为底的对数
#scipy.einsum()
#scipy.logn()
