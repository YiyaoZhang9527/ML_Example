# coding:utf-8
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as sci
from pylab import *
from time import strftime, localtime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from random import randrange, randint
import jieba
from class_filetools_module import mkdir
from os import getcwd

dirpath = getcwd() + '/plot_file/statistic_modules.py/'
mkdir(dirpath).__repr__()



# TODO : 这是一个傻瓜式的分析建模包 ^-^

def MyVectorDistribution(vector, title='some numbers'):
    '''
    函数名: 一维向量分布图 <-> One-dimensional vector distrbution
    :param vector: -> 向量
    :param title:
    :return: ->  作图 <-- > One-dimensional vector distrbution
    '''
    # python 原始数据结构版本
    if len(vector) > 1 and len(vector[0]) == 1:
        # mpl.rcParams['font.sans-serif'] = ['SimHei']
        # plot中参数的含义分别是横轴值，纵轴值，颜色，透明度和标签
        plt.plot(vector, 'ro-', color='b', alpha=0.8, label=title)
        # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
        plt.legend(loc="upper right")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(dirpath +str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')  # 保存该图片
        plt.show()
        plt.close()
    elif len(vector) > 1:
        try:
            vector[0][0][0].__repr__()
        except TypeError:
            try:
                plt.plot(vector, 'ro-', color='b', alpha=0.8, label=title)
                plt.legend(loc="upper right")
                plt.xlabel('X')
                plt.ylabel('Y')
                [exec("plt.plot(" + str(i) + ", 'ro-', color='b', alpha=0.8, label=title)") for i in vector]
                plt.show()
                plt.close()
            except NameError as Error:
                print(Error)


def MyMedian(vector):
    '''
    函数名：中位数
    :param vector: ->list
    :return:
    '''
    length = len(vector)
    return length % 2 == 0 and sorted(vector)[int(length / 2) - 1:int(length / 2 + 1)] or sorted(vector)[
        int(length / 2)]


def MyMode(vector):
    '''
    函数名 : 众数
    '''
    count = {i: vector.count(i) for i in set(vector)}
    return {i: j for i, j in count.items() if j == max(count.values())}


def MySampleVariance(vector):
    '''
    函数名 : 样本方差
    作用 : 展示数据的离散趋势，但是因为是**2所以比标准差要大

    公式: sum（（向量例每一项 - 平均数）** 2 ）/ （向量长度-1）
    python 数据结构计算：
    sum([((sum(vector)/len(vector))-i) **2 for i in vector])/(len(vector)-1)

    意义:
    在概率论和数理统计中，方差（英文Variance）用来度量随机变
    量和其数学期望（即均值）之间的偏离程度。在许多实际问题中，研究
    随机变量和均值之间的偏离程度有着很重要的意义。说明数据的变异性。

    :param vector:
    :return:
    '''
    npverctor = np.array(vector)
    return np.sum((np.mean(vector) - npverctor) ** 2) / (len(vector) - 1)


def MyVariance(vector):
    '''
    函数名 : 总体方差
    作用 : 展示数据的离散趋势，但是因为是**2所以比标准差要大，便于观测微小差异，和少量离群值

    公式: sum（（向量例每一项 - 平均数）** 2 ）/ （向量长度-1）
    python 数据结构计算：
    sum([((sum(vector)/len(vector))-i) **2 for i in vector])/(len(vector)-1)

    意义:
    在概率论和数理统计中，方差（英文Variance）用来度量随机变
    量和其数学期望（即均值）之间的偏离程度。在许多实际问题中，研究
    随机变量和均值之间的偏离程度有着很重要的意义。说明数据的变异性。
    单位是：平方

    :param vector:
    :return:
    '''
    npverctor = np.array(vector)
    return np.sum((np.mean(vector) - npverctor) ** 2) / len(vector)


def Mycovariance(vector1, vector2):
    '''
    函数名 : 协方差
    作用 : 证实两个向量之间连续的相关性

    公式 : COV（X，Y）=E［（X-E（X））（Y-E（Y））］
    如果有X,Y两个变量，每个时刻的“X值与其均值之差”乘以“Y值与其均值之差”
    得到一个乘积，再对这每时刻的乘积求和并求出均值，即为协方差。

    意义:
    在概率论和统计学中，协方差用于衡量两个变量的总体误差。而方差是协方差的一种特殊情
    况，即当两个变量是相同的情况。可以通俗的理解为：两个变量在变化过程中是否同向变化
    ？还是反方向变化？同向或反向程度如何？你变大，同时我也变大，说明两个变量是同向变
    化的，这是协方差就是正的。你变大，同时我变小，说明两个变量是反向变化的，这时协方
    差就是负的。如果我是自然人，而你是太阳，那么两者没有相关关系，这时协方差是0。从数
    值来看，协方差的数值越大，两个变量同向程度也就越大，反之亦然。可以看出来，协方差
    代表了两个变量之间的是否同时偏离均值，和偏离的方向是相同还是相反。

    url:https://blog.csdn.net/u013164612/article/details/80692569
    :param vector1:
    :param vector2:
    :return:
    '''
    npvector1, npvector2 = np.array(vector1), np.array(vector2)
    return np.mean((npvector1 - np.mean(npvector1)) * (npvector2 - np.mean(npvector2)))


def MySampleStdDev(vector, tool='python'):
    '''
    函数名: 样本标准差 <-> Sample_standard_deviation
    作用 : 样本标准差是方差的平方根，所以则更为说明了实际的数据离散情况，标准差越小，
    数据距离均值就越小，也说明数据的集中趋势越强，否则则说明数据的集中趋势越弱，但是
    因为是样本，所以会存在与大于总体实际情况，所以要减去1

    可接受数据类型 : list,tuple,pd.core.series.Series,np.ndarray

    计算方法 : 将数据的每一个点与均值之间的差值计算出来
    （正负勿论,平方运算本来就会消除数值的负号，所以正负无需在意）
    再一一做2次方运算求出面积，并且把所有面积求和，最后再除以样本的
    个数，就能得到样本方差。


    公式意义 :
    样本标准差是样本与均值的偏离,计算方法特点除以n-1,原因：用样本方差估算总
    体方差总会比总体方差要小，所以要做无偏修正

    基本限制 :

    延展阅读 :

    :param vector: -> 向量
    :param tool ->
    :return: ->
    '''

    if vector == None:
        return 'The vector is missing one , please check data .'

    elif tool == 'python' and isinstance(vector, (list, tuple)) == True:
        # python 原始数据结构版本
        return (sum([(sum(vector) / len(vector) - i) ** 2 for i in vector]) / len(vector) - 1) ** 0.5

    elif tool == 'numpy' and isinstance(vector, (list, tuple, np.ndarray, pd.core.series.Series)) == False:
        # numpy numpy的 np.std() 是总体标准差，所以还是要手撕
        npvector = np.array(vector)
        return np.sqrt(np.sum(np.square(npvector - np.mean(npvector))) / len(npvector) - 1)
        # np.std(np.array(vector))
    else:
        return 'Incoming vector type error : \n ' \
               'must -> list , tuple , numpy.ndarray , pandas.Series '  # from scipy


def MyPopulationStdDev(vector, tool='python'):
    '''
    函数名: 总体标准差 <-> Population standard deviation
    作用 : 标准差是方差的平方根，所以则更为说明了实际的数据离散情况，标准差越小，
    数据距离均值就越小，也说明数据的集中趋势越强，否则则说明数据的集中趋势越弱，

    可接受数据类型 : list,tuple,pd.core.series.Series,np.ndarray

    计算方法 :

    公式意义 :

    基本限制 :

    延展阅读 :

    :param vector: ->
    :return: ->
    '''

    if vector == None:
        return 'The vector is missing one , please check data .'

    elif tool == 'python' and isinstance(vector, (list, tuple)) == True:
        # python原始数据结构版本
        return (sum([(sum(vector) / len(vector) - i) ** 2 for i in vector]) / len(vector)) ** 0.5

    elif tool == 'numpy' and isinstance(vector, (list, tuple, np.ndarray, pd.core.series.Series)) == True:
        # numpy 自带函数
        # numpy 这里有个神坑，见说明
        # url:https://blog.csdn.net/zbq_tt5/article/details/100054087
        return np.std(np.array(vector))
    else:
        return 'Incoming vector type error : \n ' \
               'must -> list , tuple , numpy.ndarray , pandas.Series '  # from scipy


# 相似度距离计算三种详细介绍 余弦相似度，皮尔逊系数，修正余弦相似度
# URL：http://wulc.me/2016/02/22/《Programming%20Collective%20Intelligence》读书笔记(2)--协同过滤/

# 以下都是相似度计算
def MyCov(vector1=None, vector2=None):
    '''
    函数名: 协方差 <-> Covariance

    可接受数据类型 : list,tuple,pd.core.series.Series,np.ndarray

    计算方法 : （向量A，B的 方差） / （向量A，B的 标准差）

    公式意义 :
    在概率论和统计学中，协方差用于衡量两个变量的总体误差。而方差是协方差
    的一种特殊情况，即当两个变量是相同的情况。
    概率论和统计学中的协方差，评估两个向量如何一起变化，通俗的说，即是否
    同时偏离均值.
    可以看出，当A、B同时大于或小于均值时，协方差为正数；当一个大于均值，
    一个小于均值时，协方差为负数。

    从直观上来看，协方差表示的是两个变量总体误差的期望。
    如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值时另
    外一个也大于自身的期望值，那么两个变量之间的协方差就是正值；如果两个
    变量的变化趋势相反，即其中一个变量大于自身的期望值时另外一个却小于自
    身的期望值，那么两个变量之间的协方差就是负值。

    基本限制 : 同向性，非布尔值

    延展阅读 :
    url : https://baike.baidu.com/item/协方差/2185936?fr=aladdin

    :param vector1:
    :param vector2:
    :return:
    '''


def Mypearson1(vector1=None
               , vector2=None
               , plotKeyword=False
               , vector1name='vector1'
               , vector2name='vector2'
               , title='Mypearson'
               ):
    '''
    函数名 : 皮尔森相关系数 <-> pearson ->定长向量的皮尔逊系数

    可接受数据类型 : list,tuple,pd.core.series.Series,np.ndarray

    计算方法 :
    r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy（即person系数）
    两个连续变量(X,Y)的pearson相关性系数(Px,y)等于 ->
    它们之间的 '协方差cov(X,Y)' 除以它们各自标准差; '乘积(σX,σY)

    公式意义 :
    系数的取值总是在-1.0到1.0之间，接近0的变量被成为无相关性，
    接近1或者-1被称为具有强相关性。
    与斜方差相比，它并不会因为常数的位置变化而得出相关性的改变.
    皮尔森相关系数是衡量线性关联性的程度，p的一个几何解释是其
    代表两个变量的取值根据均值,集中后构成的向量之间夹角的余弦，
    <***> Pearson相关系数常用于基于用户的推荐系统，比其他对比
    用户的方法更胜一筹。（相对的，在基于物品的推荐系统中，常使
    用余弦相似度方法。）

    基本限制:
    两个向量的长度必须是一致的,计算的是总体标准差，请注意选择样本，但是
    不鄙视同向的

    参数说明:
    :param vector1: -> 向量A
    :param vector2: -> 向量B
    :param plotKeyword: -> 是否画图，是则选择 True 否则选择 False
    :param vector1name: -> 自定义的变量名
    :param vector2name: -> 自定义的变量名
    :return: -> 皮尔逊相关系数

    Matlab中代码 ： cor = corr(Matrix,'type','Pearson')

    延展阅读1 :
    Pearson 相关系数是用协方差除以两个变量的标准差得到的，虽然协
    方差能反映两个随机变量的相关程度（协方差大于0的时候表示两者正相关，
    小于0的时候表示两者负相关），但其数值上受量纲的影响很大，不能简单
    地从协方差的数值大小给出变量相关程度的判断。为了消除这种量纲的影响
    ，于是就有了相关系数的概念。

    延展阅读2 :
    量纲（dimension）也叫因次，是指物理量固有的、可度量的物理属性。
    一个物理量是由自身的物理属性（量纲）和为度量物理属性而规定的量度单
    位两个因素构成的。每一个物理量都只有一个量纲，不以人的意志为转移；
    每一个量纲下的量度单位（量度标准）是人为定义的，因度量衡的标准和尺
    度而异。量纲通常用一个表示该物理量的罗马正体大写字母表示。量纲分
    为基本量纲和导出量纲。国际单位制（SI）规定了七个基本物理量，相对
    应的是七个基本量纲；其他任何物理量的量纲均可以通过这些基本量纲导
    出，称为导出量纲。导出量纲与七个基本量纲一定满足对数线性组合关系。
    如果一个物理量可以用一个纯实数来衡量（如应变），则这个物理量为无量
    纲（又称“量纲一”或“纯数”）。无量纲量是量纲分析和相似理论的基础。
    量纲一的量可以进行各种超越运算；有量纲的量可以进行乘法运算和对数
    运算，规定有量纲量对数的量纲为量纲一；相同量纲的量可以进行加法运算
    。量纲的运算必须满足量纲和谐原理。

    算法说明1 : url :https://blog.csdn.net/huangfei711/article/details/78456165
    #算法说明1偏于学术
    算法说明2 : url ：https://blog.csdn.net/AlexMerer/article/details/74908435
    #算法说明2便于理解基本概念

    使用代码示例：
    vector1 = [2, 7, 18, 88, 157 , 90, 177, 570]
    vector2 = [3, 5, 15, 90, 180, 88, 160, 580]
    print(pearson(vector1,vector2))
    '''
    if vector1 == None and vector2 == None:
        return 'No vector input'

    elif vector1 == None or vector2 == None:
        return 'The vector is missing one , please check data .'

    elif len(vector1) != len(vector2):
        return 'The vector length is inconsistent, please check .'

    elif isinstance(vector1, (list, tuple
                              , np.ndarray, pd.core.series.Series)) == False \
            and isinstance(vector1, (list, tuple
                                     , np.ndarray, pd.core.series.Series)) == False:

        return 'Incoming vector type error : \n ' \
               'must -> list , tuple , numpy.ndarray , pandas.Series .'  # from scipy

    elif isinstance(vector1, (np.ndarray
                              , pd.core.series.Series)) == True \
            and isinstance(vector1, (np.ndarray
                                     , pd.core.series.Series)) == True:

        if plotKeyword == True:
            sns.jointplot(x=vector1name
                          , y=vector2name
                          , data=pd.DataFrame({
                    vector1name: vector1  # 变量1
                    , vector2name: vector2})  # 变量2
                          , color='c'  # 画图颜色：c 青色 ，b 蓝色 ，r红色 ， y黄色 ，
                          , kind="reg"
                          , height=8  # 图表大小(自动调整为正方形))
                          , ratio=5  # 散点图与布局图高度比，整型
                          )
            plt.savefig(dirpath +str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')  # 保存该图片
            plt.show()
            plt.close()
            return sci.pearsonr(vector1, vector2)

        else:
            return sci.pearsonr(vector1, vector2)
    else:
        n = len(vector1)  # 计算向量长度
        # simple sums
        sum1 = sum(float(vector1[i]) for i in range(n))  # 向量转化浮点数
        sum2 = sum(float(vector2[i]) for i in range(n))
        # sum up the squares
        '''pow() 是计算数值的次方
        方法返回 xy（x的y次方）的值 ，说明链接
        '''
        '''计算：求向量每一个值的平方和'''
        sum1_pow = sum([pow(v, 2.0) for v in vector1])
        sum2_pow = sum([pow(v, 2.0) for v in vector2])
        # sum up the products
        '''计算:向量每一个值的一一相乘'''
        p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
        # 分子num，分母den
        ''' 计算协方差 '''
        num = p_sum - (sum1 * sum2 / n)
        ''' 计算标准差'''
        den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))

        if den == 0:
            return 0.0

        elif plotKeyword == False:
            return num / den

        elif plotKeyword == True:
            sns.jointplot(x=vector1name
                          , y=vector2name
                          , data=pd.DataFrame({
                    vector1name: vector1  # 变量1
                    , vector2name: vector2})  # 变量2
                          , color='m'  # 画图颜色：c 青色 ，b 蓝色 ，r红色 ， y黄色 ，m 品红色 ，k 黑色 ，w 白色
                          , kind="reg"
                          , height=8  # 图表大小(自动调整为正方形))
                          , ratio=5  # 散点图与布局图高度比，整型
                          )
            plt.savefig(dirpath +str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')  # 保存该图片
            plt.show()
            plt.close()
            return num / den


def linearRegression(vector1=None
                     , vector2=None
                     , plotKeyword=False
                     , vector1name='vector1'
                     , vector2name='vector2'
                     , title='linearRegression'):
    '''
    函数名 : 一元线性回归

    计算公式 :

    公式意义 :

    结果意义 : 斜率就是协方差，但是截距是向量在二维坐标系上的基的y轴，

    条件限制 :

    拓展阅读 :

    :param x_data: -> 样本点x坐标的list
    :param y_data: -> 样本点y坐标的list
    :return: ->  回归方程的斜率e1与截距e0

    '''
    if vector1 != None and vector2 != None and isinstance(vector1, (list, tuple)) and isinstance(vector2,
                                                                                                 (list, tuple)):
        # list->np.array,list(x),list(y)
        '''数据类型转npdarray'''
        npvector1 = np.array(vector1)
        npvector2 = np.array(vector2)
        '''分别计算xy向量的平均值'''
        npvector1_mean = np.mean(npvector1)  # list(x)的平均值
        npvector2_mean = np.mean(npvector2)  # list(y)'s mean
        '''计算两个向量中所有数据点距离向量均值的距离'''
        residual1 = npvector1 - npvector1_mean
        residual2 = npvector2 - npvector2_mean
        '''查一下为什么'''
        var_1 = residual1 * residual2
        var_2 = residual1 * residual1
        '''回归线的斜率'''
        slope = round((np.sum(var_1) / np.sum(var_2)) * 10) / 10  # slope
        '''回归线的截距'''
        intercept = round((npvector2_mean - slope * npvector2_mean) * 10) / 10  # intercept
        # Drawing
        # print(['slope:',slope,'intercept:',intercept])
        '''作图模块'''
        if plotKeyword == True:
            min_X = np.min(vector1)
            max_X = np.max(vector1)
            x_plot = [min_X, max_X]
            y_plot = [min_X * slope + intercept, max_X * slope + intercept]
            plt.scatter(vector1, vector2, label='root data', color='k', s=5)
            plt.plot(x_plot, y_plot, label='regression line')
            plt.xlabel(vector1name)
            plt.ylabel(vector2name)
            plt.legend()
            plt.savefig(dirpath +str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')  # 保存该图片
            plt.show()
            plt.close()
            return ['slope:', slope, 'intercept:', intercept]
        else:
            return ['slope:', slope, 'intercept:', intercept]
    else:
        return None

def as_num(x):
    '''
    科学计数转常数百分比显示
    :param x:
    :return:
    '''
    y = '{:.10000f}'.format(x) # 5f表示保留5位小数点的float型
    num = []
    results = []
    for i in y[::-1]:
        num.append(i)
        test = sum(list(map(lambda x : ord(x) in list(range(48,57)) and int(x) or 0 , num )))
        if test != 0:
            results.append(i)
    return 'P = '+''.join(results[::-1])+'%'

def factorial0(n):
    '''
    递归阶乘函数
    :param n:
    :return:
    '''
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def factorial(a):
    '''
    阶乘函数
    https://www.zhihu.com/question/36214010/answer/208718886 阅读衍生
    :param a: 输入数
    :return:
    '''
    num = 1
    if a < 0:
        print('负数没有阶乘！')
    elif a == 0:
        print('0的阶乘为1！')
    else :
        for i in range(1,a + 1):
            num *= i
    return num

np.set_printoptions(suppress=True)

def binomial_distribution(P,k,n):
    '''
    二项式函数
    :param P: 初始概率
    :param k: 成功的组合数
    :param n: 组合基数
    :return: 二项分布计算结果
    '''
    reverse_of_P = 1-P #反向的概率
    Ps = (P ** k) #正向的概率和
    reverse_of_Ps = np.power(reverse_of_P, (n - k)) # 反项概率和
    Cnk = factorial(n)/(factorial(n-k)*factorial(k)) # 二项系数
    expr = Cnk * (Ps * reverse_of_Ps) #条件成功概率
    return expr , as_num(expr)

def DiscreteYebes():
    '''
    函数名 : 贝叶斯算法

    计算公式 :
    P 是概率
    检验的公式为:
    在B条件成立下，事件A的概率也成立,此时，事件A的概率 =（B的概率*A的概率）/ B的概率
    伪代码:if （B条件 == True and A条件 == True）== True:
               A 的概率 =  (B 的概率 * A 的概率) / B 的概率 if A 条件 == True else None

    P(Trueset) 是条件1为真的概率 -> （先验概率）
    P(Flaseset) 是条件1为假的概率 -> （先验概率）
    P(Trueset(True)) 是条件1为真的集合里真的概率 -> （后验概率）
    P(Trueset(False)) 是条件1为真的集合里假的概率 ->（后验概率）
    更新检验的公式为：
    P(Trueset) * P(Trueset|Tureset(Ture)) / P(Falseset) * P(True|Trueset(False))

    公式意义 :

    贝叶斯定理常常被用来更新和检验假设
    >例1 推断检验假设:
    你听说一个人是乳腺癌，但是并不知道此人是男还是女,所以可以通过数据计算得到此人性别的可能性：
    某一个男性并且患有乳腺癌的概率 =  (男性并且患有乳腺癌的概率 * 是男性的概率) / 乳腺癌的概率
    当获得男性患有乳腺癌的概率后，就可以用来更新判断此人是男性的概率，因为男性获得乳腺癌的概率
    可以当作此人是男性的概率的等价。

    >例2 概率更新:
    判断一个人是不是星球大战的粉丝
    过程1：
        初始信息:
        人群中星球大战的粉丝=60%
        非星球大战的粉丝=40%
        判断1，某人看过星球大战概率是60%
    如果判断1成立:
        更新信息1:
        最新看过星球大战的人里粉丝概率是99%
        不是粉丝而因为其他原因去看的人概率是5%
    判断2，某事是星球大战粉丝的概率为:
        -> 99% / 5% = 198% <- (这里的198%为叶贝斯因子，代表通过学习获得的概率更新)
    总结:
        此时的后验概率：
        (60%/40%) * (99%/5%) =>  1.5 * 1.98 => 2.97


    条件限制 :

    拓展阅读 :
    视频URL：https://www.bilibili.com/video/av55293534?from=search&seid=10696006316636312672
    :return:
    '''


'''各种相似度计算的python实现'''


def euclidean(p, q):
    '''
    函数名 ：欧几里德距离：

    几个数据集之间的相似度一般是基于每对对象间的距离计算。最
    常用的当然是欧几里德距离，

    基本限制 : 高度相关很适合，但是计算机计算cpu开销过大，对
    数值太敏感，在astart算法力可以用闵氏距离替代

    :param p:
    :param q:
    :return:

    代码示例 :
    p = [1, 3, 2, 3, 4, 3]
    q = [1, 3, 4, 3, 2, 3, 4, 3]
    print(euclidean(p, q))
    '''
    # 如果两数据集数目不同，计算两者之间都对应有的数
    same = 0
    for i in p:
        if i in q:
            same += 1
    # 计算欧几里德距离,并将其标准化
    e = sum([(p[i] - q[i]) ** 2 for i in range(same)])
    return 1 / (1 + e ** .5)


def Mypearson2(p, q):
    '''
    不定长向量的：皮尔逊相似度，计算两者共同有的元素
    适合用来筛选文本文字相似性
    数学意义上与余弦相似度等价，
    :param p:
    :param q:
    :return:
    '''

    # 只计算两者共同有的,元素必须同位置
    same = 0
    for i in p:
        if i in q:
            same += 1
    n = same
    # 分别求p，q的和
    sumx = sum([p[i] for i in range(n)])
    sumy = sum([q[i] for i in range(n)])
    # 分别求出p，q的平方和
    sumxsq = sum([p[i] ** 2 for i in range(n)])
    sumysq = sum([q[i] ** 2 for i in range(n)])
    # 求出p，q的乘积和
    sumxy = sum([p[i] * q[i] for i in range(n)])
    # print sumxy
    # 求出pearson相关系数
    up = sumxy - sumx * sumy / n
    down = ((sumxsq - pow(sumxsq, 2) / n) * (sumysq - pow(sumysq, 2) / n)) ** .5
    # 若down为零则不能计算，return 0
    if down == 0: return 0
    r = up / down
    return r


def manhattan(p, q):
    '''
    # 曼哈顿距离：-> Astart算法中就可以替代欧式距离
    :param p:
    :param q:
    :return:
    '''
    # 只计算两者共同有的
    same = 0
    for i in p:
        if i in q:
            same += 1
    # 计算曼哈顿距离
    n = same
    vals = range(n)
    distance = sum(abs(p[i] - q[i]) for i in vals)
    return distance


def Jaccrad(model, reference):  # terms_reference为源句子，terms_model为候选句子
    '''
    # 计算jaccard系数
    :param model:
    :param reference:
    :return:

    延展阅读
    url：https://blog.csdn.net/ice110956/article/details/28917041
    '''
    # 果然还是要用到jieba分词
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = set(terms_reference)  # 去重；如果不需要就改为list
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient


def tanimoto(p, q):
    '''
    函数名 : 基于谷本系数计算相似度
    Tanimoto系数和皮尔逊系数还要欧氏距离一样可以用来判断两个数
    据的相关程度。Tanimoto系数可以表示为公式 : 两个集合的交比
    上两个集合的并

    延展阅读 :
    url:https://blog.csdn.net/ice110956/article/details/28917041
    :param p:
    :param q:
    :return:
    '''


'''
python 运算符模块介绍 : https://www.cnblogs.com/who-care/p/9839058.html

10分钟统计速成班 youtube url : https://www.youtube.com/playlist?list=PL8dPuuaLjXtNM_Y-bUAhblSAdWRnmBUcr
微积分的本质是什么 url : https://www.bilibili.com/video/av11041729?from=search&seid=14868484735832423304
导数的讲解 URL： https://www.zhihu.com/question/355907163/answer/896501095

要实现的和理解的算法 :
1。
斜率 --> 视频url：https://www.bilibili.com/video/av11041729?from=search&seid=14868484735832423304
截距
残差
最小二乘法
一元线性回归  url要翻墙 ：https://www.youtube.com/watch?v=WWqE7YHR4Jc
多元线性回归
逻辑回归 
2。
数据归一化 公式 --> newValue = (oldValue – min) / (max – min) 视频：https://www.cnblogs.com/netuml/p/5721251.html
KN
3.
关联算法 
4.
粗糙集
5.
短文本算法
6.
二项分布
7.
泊松分布
几何分布 ：https://www.jianshu.com/p/66c690fabe98，视频：https://www.bilibili.com/video/av20624185?p=17
8
7种距离计算
9.
众数
中位数
四分位法
AStart
推荐算法的讲解：url ： http://wulc.me/2016/02/22/《Programming%20Collective%20Intelligence》读书笔记(2)--协同过滤/
'''

if __name__ == '__main__':
    a = [randint(1, 100) for i in range(10000)]
    b = list(reversed(a))
    print(a, '\n', b)
    print(linearRegression(vector1=a, vector2=b, plotKeyword=True))

    '''
    #欧几里得距离
    p = [1, 3, 2, 3, 4, 3]
    q = [1, 3, 4, 3, 2, 3, 4, 3]
    print(euclidean(p, q))
    '''
    '''
    #皮尔逊相关系数测试
    #可以不定长
    p = [1, 3, 2, 3, 4, 3]
    q = [1, 3, 4, 3, 2, 3, 4, 3]
    print(Mypearson2(p, q))
    '''
    '''
    a = "香农在信息论中提出的信息熵定义为自信息的期望"
    b = "信息熵作为自信息的期望"
    jaccard_coefficient = Jaccrad(a, b)
    print(jaccard_coefficient)
    '''
    '''
    a = "香农在信息论中提出的信息熵定义为自信息的期望"
    b = "信息熵作为自信息的期望"
    ax = [ord(i) for i in a]
    by = [ord(i) for i in b]
    print(Mypearson2(ax,by))
    '''
    '''
    p = [1, 3, 2, 3, 4, 3]
    q = [1, 3, 4, 3, 2, 3, 4, 3]
    print(manhattan(p, q))
    '''
    '''
    #皮尔逊相关系数测试
    #必须定长
    #vector1 = [2, 7, 18, 88, 157 , 90, 177, 570]
    #vector2 = [3, 5, 15, 90, 180, 88, 160, 580]

    for c in range(10000):
        vector1 = [randrange(-10000, 10000) for i in range(2000)]
        vector2 = [randrange(-10000, 10000) for i in range(2000)]
        p = Mypearson(vector1=vector1,vector2=vector2,vector1name='TT',vector2name='ZZ')
        print(c,p)
        if p > 0.3:
            print(c,p,vector1,vector2)
            '''

    '''
    vector = [2, 1, 7, 18, 88, 157, 90, 177, 570]
    print(MyPopulationStdDev(vector))
    print(MyPopulationStdDev(vector,tool='numpy'))
    '''
    '''
    vector = [2, 1, 7, 18, 88, 157, 90, 177, 570]
    print(MySampleStdDev(vector))
    print(MySampleStdDev(vector))
    '''
    '''
    vector = [2, 1, 7, 18, 88, 157, 90, 177, 570]
    MyVectorDistribution(vector=vector,title='suibian').__doc__.__repr__()
    '''
