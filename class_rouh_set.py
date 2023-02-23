
import numpy as np
from scipy.special import perm
from functools import reduce 
from itertools import permutations , combinations
# TODO 这里是粗糙集算法的函数包


path = '/Users/manmanzhang/Library/Mobile Documents/com~apple~CloudDocs/MyProject/InferenceSystem/src/I5_algorithm/setcase.csv'
data = np.loadtxt(path,dtype=str,delimiter=',')
'''
data = np.array([['ID', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜'],
       ['1', '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
       ['2', '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
       ['3', '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
       ['4', '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
       ['5', '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
       ['6', '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
       ['7', '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
       ['8', '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
       ['9', '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
       ['10', '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
       ['11', '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
       ['12', '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
       ['13', '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
       ['14', '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
       ['15', '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
       ['16', '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
       ['17', '青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']], dtype='<U2')
'''
print(data)


def Theory_of_domain(information_table):
    '''
    论域
    :return: 
    '''
    return information_table[1:,0:1]

#Theory_of_domain(data)

def Decision_attribute(information_table):
    '''
    决策属性
    :param information_table: 
    :return: 
    '''
    return information_table[1:,-1:]

#Decision_attribute(data)



def Condition_attributes(information_table):
    '''
    条件属性
    :param information_table: 
    :return: 
    '''
    return information_table[1:,1:-1]

#Condition_attributes(data)



def Quotient_set(information_table):
    '''
    信息表的所有商集
    :return: 
    '''
    #去重
    Condition_Attributes = Condition_attributes(information_table)
    duplicate_removal = list(map(lambda x: np.array(eval(x)),set([repr(list(i)) for i in Condition_Attributes])))
    return np.array(duplicate_removal),'difference:',len(Condition_Attributes)-len(duplicate_removal)

#Quotient_set(data)

def Equivalence_class(X,information_table):
    '''
    等价类
    :return: 
    '''
    expr = np.array([np.array([column for column in information_table if element in column]) for element  in X])
    join_table = np.concatenate(tuple(expr))
    all_table_index = sorted([eval(i) for i in np.unique(join_table[:,0:1])])
    all_table = information_table[all_table_index]
    results = '组合表等价类:',all_table ,'组合表等价类下标:',all_table_index ,'分表等价类:',expr ,'分表等价类下标:', [np.array([eval(j[0]) for j in i[:,0:1]]) for i in expr]
    return results
        

#Equivalence_class(['青绿', '蜷缩'],data)
# TODO 求出所有信息表的组合方式

def spread(arg):
    '''
    广播函数
    :param arg:
    :return:
    '''
    ret = []
    for i in arg:
        if isinstance(i, (list,tuple)):
            ret.extend(i)  # 列表末尾追加 等同于 ret += i
        else:
            ret.append(i)
    return ret


def deep_flatten(lst):
    '''
    深度平展
    :param lst:
    :return:
    '''
    result = []
    result.extend(
        spread(list(map(lambda x: deep_flatten(x) if type(x) in (list,tuple) else x, lst))))
    return result

def my_combination(x):
    '''
    多组组合
    :param x: 
    :param code: 
    :return: 
    '''
    return reduce(lambda x, y: [deep_flatten([i,j]) for i in x for j in y], x)

def combination_module_of_information_table(information_table):
    '''
    求出所有信息表的组合方式
    '''
    condition = Condition_attributes(information_table)
    #列出信息表内所有信息元素的二维结构展示
    columns = [list(np.unique(condition[:,i:i+1])) for i in range(len(condition[0]))]
    #计算出所有组合方式
    #counts = list(map(lambda x : [perm(len(x),i) for i in range(1,len(x)+1)] ,columns))
    #print(counts)
    return np.array([i for i in map(lambda x : np.array(x),my_combination(columns))])

#combination_module_of_information_table(data)



def Missing_combination_calculation(information_table):
    '''
    求出所有信息表内不完全组合的计算
    :return: 
    '''
    condition = Condition_attributes(information_table)
    #列出信息表内所有信息元素的二维结构展示
    columns = [list(np.unique(condition[:,i:i+1])) + [None] for i in range(len(condition[0]))]
    #计算出所有组合方式
    temporary =  np.array([np.array([j for j in i if j != None]) for i in my_combination(columns) if  [j for j in i if j != None] != []])
    return temporary
    
#Missing_combination_calculation(data)


# TODO 这一部分知识为了求等价类 

def Classification_of_decision(information_table):
    '''
    决策分类
    :param information_table: 
    :return: 
    '''
    index_1 = information_table[:,-1:].reshape(-1)
    get_index_defference = list(np.unique(index_1))
    return np.array([index_1[index_1==i] for i in get_index_defference])

#Classification_of_decision(data)



def Enumerates_the_quotient_that_appears(information_table):
    '''
    枚举集合所有出现的商集
    :param information_table: 
    :return: 
    '''
    
    condition = Condition_attributes(information_table)
    slice_range = condition[0].size

    slicing_rule = [list(combinations(list(range(slice_range)),i)) for i in range(1,slice_range)]
    flatten_function = lambda data : [i for j in range(len(data)) for i in data[j]]
    flatten = flatten_function(slicing_rule)
    all_the_slices = [[[condition[y][j] for j in i] for i in flatten] for y in range(len(condition))]
    return np.array(list(map(lambda z : np.array(z),list(map(lambda y : eval(y),set(list(map(lambda x : str(x) ,flatten_function(all_the_slices)))))))))
    
#Enumerates_the_quotient_that_appears(data)


def Equivalence_class_of_all(information_table):
    '''
    求出所有等价类->等价类就是相同条件的集合
    :param information_table: 
    :return: 
    '''
    condition = Condition_attributes(information_table)
    results1 = []
    enumerates_q = Enumerates_the_quotient_that_appears(information_table)
    contrasts = np.unique(condition)
    index_dict = {i:j for i,j in zip(contrasts,range(len(contrasts)))}
    
    for i in enumerates_q:
        exec(''.join(i)+' = []') 

    for a in enumerates_q:
        for b in condition:
            p = np.all(np.isin(a,b))
            if p:
                exec(''.join(a)+'.append({"tag":a,"data":b})')   
            else:
                pass
    data_set = []
    for i in enumerates_q:
        data_set.append(np.array(eval(''.join(i))))
    split_func = lambda keyword :np.array([i[0][keyword] for i in data_set])
    datas = split_func("data")
    tags = split_func("tag")
    indexs = np.array([np.array([index_dict[j] for j in i]) for i in datas])

    return {'tags':tags,'datas':datas,'indexs':indexs,'contrasts':contrasts,'data_set':data_set}

#Equivalence_class_of_all(data)['data_set'][0]

def Reduction(information_table):
    '''
    集合约简
    :return:
    '''
    equivalence = Equivalence_class_of_all(information_table)
    tags , datas ,indexs ,contrasts = equivalence['tags'] ,equivalence['datas'] ,equivalence['indexs'] ,equivalence['contrasts']
    return [list(combinations(list(range(len(indexs))),i)) for i in range(1,len(indexs)+1)]



print(Reduction(data))
# %%
def PosR(X):
    '''
    上近似
    :return: 
    '''
    


# %%
def negR(X):
    '''
    下近似
    :return: 
    '''
    


# %%
def bnR(X):
    '''
    X的R边界线
    :param X: 
    :return: 
    '''


# %%
def Positive_domain():
    '''
    正域
    :return: 
    '''


# %%
def Negative_domain():
    '''
    负域
    :return: 
    '''

