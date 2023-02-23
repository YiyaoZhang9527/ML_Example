from pylab import *
from time import strftime, localtime
from class_filetools_module import mkdir
from os import getcwd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dirpath = getcwd() + '/plot_file/simple_linear_regression/'
mkdir(dirpath).__repr__()

# ：TODO 这是一元线性回归函数，采用最小二乘法

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
        '''过渡变量方便后面计算代码缩短'''
        var_1 = residual1 * residual2
        var_2 = residual1 * residual1
        '''回归线的斜率'''
        slope = round((np.sum(var_1) / np.sum(var_2)) * 10) / 10  # slope
        '''回归线的截距'''
        intercept = round((npvector2_mean - slope * npvector1_mean) * 10) / 10  # intercept
        # Drawing
        # print(['slope:',slope,'intercept:',intercept])
        '''作图模块'''
        if plotKeyword == True:
            min_X = np.min(vector1)
            max_X = np.max(vector1)
            x_plot = [min_X, max_X]
            y_plot = [min_X * slope + intercept, max_X * slope + intercept]
            plt.scatter(vector1, vector2, label='root data', color='k', s=5)
            plt.plot(x_plot
                     , np.array(y_plot) + intercept  # y的坐标要-截距
                     , label='regression line')  # 这是回归线
            plt.xlabel(vector1name)
            plt.ylabel(vector2name)
            plt.legend()
            plt.savefig(dirpath+str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')  # 保存该图片
            plt.show()
            plt.close()
            return ['slope:', slope, 'intercept:', intercept]
        else:
            return ['slope:', slope, 'intercept:', intercept]
    else:
        return None


if __name__ == '__main__':
    sure = [15, 39, 60, 70, 106, 152]  # 确诊人数
    m = [310, 664, 1023, 1521, 1791, 410]  # 医学观察人数
    a = [randint(1, 100) for i in range(10000)]
    '''
    b = list(reversed(a))
    print(a,'\n',b)
    print(linearRegression(vector1=a,vector2=b,plotKeyword=True))
    '''
    print(linearRegression(vector1=[i for i in range(len(sure))], vector2=sure, plotKeyword=True))
