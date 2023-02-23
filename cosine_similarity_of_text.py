import jieba
import jieba.analyse
import math
from scipy import spatial
import numpy as np
from collections import Counter

class cosine_similarity_of_text:

    def __init__(self,strings1=None,strings2=None,model='fitnumpy'):
        self.strings1 , self.strings2 = strings1 , strings2
        self.strings1_cut, self.strings2_cut = self.cut(strings1), self.cut(strings2)
        self.strings1_cut_code ,self.strings2_cut_code = self.cut_code(self.strings1_cut) , self.cut_code(self.strings2_cut)
        self.word_set = self.word_set(self.strings1_cut,self.strings2_cut)
        self.constant_full_word_dict = self.word_dict(self.word_set)
        self.frequency_of_word1 = self.frequency_of_word(self.strings1_cut,self.strings1_cut_code,self.constant_full_word_dict)
        self.frequency_of_word2 = self.frequency_of_word(self.strings2_cut,self.strings2_cut_code,self.constant_full_word_dict)
        if model == 'numpy':
            self.fit = self.full_npcosine(np.array(self.frequency_of_word1),np.array(self.frequency_of_word2))
        elif model == 'python':
            self.fit =self.full_pycosine(self.frequency_of_word1, self.frequency_of_word2).__next__()
        self.strings1_tag = self.words_tag(self.strings1)
        self.strings2_tag = self.words_tag(self.strings2)
        self.word_vec1 ,self.word_vec2 =  self.word_vec(self.strings1_tag)  ,self.word_vec(self.strings2_tag)
        self.words_2_vec_dict = self.word_dict(set([word[0] for word in self.strings1_tag])|set([word[0] for word in self.strings2_tag]))


    def cut(self,strings):
        '''
        分词
        :param strings:
        :return:
        '''
        return [i for i in jieba.cut(strings, cut_all=True) if i != '']

    def word_set(self,strings1_cut,strings2_cut):
        '''
        并集词汇表
        :return:
        '''
        return set(strings1_cut) | (set(strings2_cut))

    def word_dict(self,stings):
        '''
        创建字典
        :return:
        '''
        return {tuple(stings)[i]: i for i in range(len(stings))}

    def cut_code(self,cut):
        '''
        单文词语编码
        :param cut:
        :return:
        '''
        return (self.constant_full_word_dict[word] for word in cut)

    def frequency_of_word(self,string_cut,string_cut_code,full_dict):
        '''
        统计词频生成词向量
        :param string_cut:
        :param string_cut_code:
        :return:
        '''
        dict_ = full_dict
        string_cut_code = string_cut_code
        string_cut_code = [0] * len(dict_)
        for word in string_cut:
            string_cut_code[dict_[word]] += 1
        return (string_cut_code)

    def full_pycosine(self,vector1,vector2):
        '''
        python数据结构计算余弦相似度
        :param vector1:
        :param vector2:
        :return:
        '''
        sum = 0
        sqrt1 = 0
        sqrt2 = 0
        for i in range(len(vector1)):
            sum += vector1[i] * vector2[i]
            sqrt1 += pow(vector1[i], 2)
            sqrt2 += pow(vector2[i], 2)
        try:
            result = yield round(float(sum) / (math.sqrt(sqrt1) * math.sqrt(sqrt2)), 2)
        except ZeroDivisionError:
            result = 0.0
        return result

    def full_npcosine(self,vector1,vector2):
        '''
        scipy方法创建ndarray结构的余弦相似度
        :param vector1:
        :param vector2:
        :return:
        '''
        return spatial.distance.cosine(vector1,vector2)

    def words_tag(self,strings):
        '''
        提取关键词
        :param strings:
        :return:
        '''
        return jieba.analyse.extract_tags(strings,withWeight=True)

    def word_vec(self,strings_tag):
        return Counter({i[0]: i[1] for i in strings_tag})




    def __del__(self):
        pass


if __name__ == '__main__':
    strings1 = '从数学上看，余弦相似度衡量的是投射到一个多维空间中的两个向量之间的夹角的余弦。当在多维空间中绘制余弦相似度时，余弦相似度体现的是每个向量的方向关系（角度），而非幅度。如果你想要幅度，则应计算欧几里德距离。'
    strings2 = '余弦相似度很有优势，因为即使两个相似的文件由于大小而在欧几里德距离上相距甚远（比如文档中出现很多次的某个词或多次观看过同一部电影的某用户），它们之间也可能具有更小的夹角。夹角越小，则相似度越高。'
    fit = cosine_similarity_of_text(strings1, strings2,model='python')
    print(fit.word_vec1)

