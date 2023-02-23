import modin.pandas as pd
import numpy as np
from class_error_model import longerror
from class_data_structure_module import deep_flatten

# TODO 这里是粗糙集算法的函数包


path = '/Users/manmanzhang/Library/Mobile Documents/com~apple~CloudDocs/MyProject/InferenceSystem/src/I5_algorithm/setcase.csv'
data = np.loadtxt(path,dtype=str,delimiter=',')


def equivalence_class1(information_table):
    '''
    组合计算出集合的所有商为2的等价类
    :param information_table: --> numpy结构的表格数据
    :return: --> 所有商为2的等价类
    '''
    ''''''
    '''求出列数'''
    column_name = len(information_table[0])
    print(information_table)
    '''信息表范围'''
    columns_indexs_range = range(1,column_name-1)
    '''组合计算出集合的所有商为2的等价类'''
    columns_indexs = [(i , j )for i in columns_indexs_range for j in columns_indexs_range if i != j]
    return [np.hstack((information_table[:,i[0]:i[0]+1],information_table[:,i[-1]:i[-1]+1])) for i in columns_indexs]

def equivalence_class2(information_table):
    print(information_table)
    first  = information_table[1:,1:2]
    index_set = list(np.unique(first))
    index_all = [index_set.index(i)+1 for i in first]
    print(index_all)
    column = [np.squeeze(information_table[1:,i:i+1]) for i in range(1,len(information_table[1])-1)]
    print(column)
    flags = [set(i) for i in column]
    print(flags)
    for i in information_table[1:,1:-1]:
        flag = [set(j) for j in flags]
        #print(flag)













def rough_set(information_table):
    #column = information_table.columns
    #data = np.array([np.array(information_table[information_table.name==i])[0] for i in information_table[column[0]]])
    #results = list(data[:,-1:])
    '''

    :param information_table:
    :return:
    '''
    '''列名称'''
    column = information_table[0]
    '''行名称'''
    name = information_table[1:,0:1]
    '''读取分类标签'''
    flag = np.unique(information_table[1:,-1:])
    '''表格信息内容'''
    data = information_table[:,1:-1]

    '''分类得到下标'''
    classification = [[[(k,w)for k,w in zip(np.unique(data[j]),[np.sum(data[j]==k) for k in np.unique(data[j])])] for j in range(len(information_table)) if i == information_table[j][-1:]] for i in set(flag)]
    '''降维分类后的list'''
    print(classification)



    '''计算结果'''

    for flag,droplist in zip(set(flag),classification):

        drop = [i for j in range(len(droplist)) for i in droplist[j]]
        index_drop0 = np.array([i[0] for i in drop])
        count_drop1 = np.array([i[-1] for i in drop])
        count_drop2 = [(k,np.sum(index_drop0==k)) for k in np.unique(index_drop0)]

        expr2 = []
        for i in drop:
            for j in count_drop2:
                if i[0]==j[0]:
                    expr2.append([i[0],i[-1]+j[-1]])
        print('*'*100)
        for i,j in reversed(sorted(expr2)):
            print(i,j)
    return expr2

if __name__ == '__main__':


    print(equivalence_class2(data))
    #Roughlogic().__repr__()

