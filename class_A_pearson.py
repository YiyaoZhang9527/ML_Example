# coding:utf-8
import seaborn as sns
import pandas as pd
import scipy.stats as sci
from pylab import plt
from time import strftime, localtime
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from os import getcwd
from class_filetools_module import mkdir
import math

dirpath = getcwd() + '/plot_file/Pearson_plot/'

mkdir(dirpath).__repr__()


# TODO : 皮尔逊相似度

def MyPearson(vector1=None
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
            plt.savefig(dirpath+str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')  # 保存该图片
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
            plt.savefig(dirpath+str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')
            plt.show()
            plt.close()
            return num / den


if __name__ == '__main__':
    # 皮尔逊相关系数测试
    # 必须定长
    sure = [440, 571, 830, 1287, 1975, 2744]
    dead = [9, 17, 25, 41, 56, 80]
    print(MyPearson(vector1=sure
                    , vector2=dead
                    , plotKeyword=True
                    , vector1name='sure'
                    , vector2name='dead'
                    , title='sure and dead'))
