import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
from class_filetools_module import mkdir
from time import strftime, localtime
import sys


dirpath = getcwd() + '/plot_file/plot_gradint_same/'
print(mkdir(dirpath))


# TODO : 长文本相似度算法，梯度绝对值相似度
def gradient_SABS(vector=None,title='gradient_SABS',plotkey=False):
    '''
    函数名：数组梯度均值相似度
    :param vector: -> int
    :return:
    '''
    if isinstance(vector, (int, float)):
        # 作图
        plt.plot([i for i in range(len(vector))], np.gradient(vector))
        plt.show()
        # 计算文字梯度均值
        return sum(np.abs(np.gradient(vector))) / len(vector)
        # 检测传入是否是数据结构
    elif isinstance(vector, (list, tuple)):
        # 计算文字梯度均值
        gradlist = [sum(np.abs(np.gradient(ver))) / len(ver) for ver in vector]
        # 作图
        for line in vector:
            plt.plot([i for i in range(len(line))], np.gradient(line))
            plt.savefig(dirpath + str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')
        plt.show()
        return gradlist
    return sum(np.abs(np.gradient(vector))) / len(vector)

# TODO : 长文本相似度算法，梯度绝对值相似度
def gradient_SABS_str(string = None,title='strings_of_gradient_SABS',plotkey=False):
    '''
    函数名：长文本字符梯度均值相似度
    :param string: -> str
    :return:
    '''
    # 检测是否为数据类型
    if isinstance(string, (str, int, float)):
        # 字符串数字化
        vector = [ord(i) for i in string]
        if plotkey == True:
            # 作图
            plt.plot([i for i in range(len(vector))], np.gradient(vector))
            plt.savefig(dirpath + str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')
            plt.show()
            # 计算文字梯度均值
            return sum(np.abs(np.gradient(vector))) / len(vector)
        elif plotkey == False:
            # 计算文字梯度均值
            return sum(np.abs(np.gradient(vector))) / len(vector)
        else:
            return None
    # 检测传入是否是数据结构
    elif isinstance(string, (list, tuple)):
        # 字符串数字化
        vector = [[ord(i) for i in s] for s in string]
        # 计算文字梯度均值
        gradlist = [sum(np.abs(np.gradient(ver))) / len(ver) for ver in vector]
        # 作图
        for line in vector:
            plt.plot([i for i in range(len(line))], np.gradient(line))
            plt.savefig(dirpath + str(title) + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.png')
        plt.show()
        return gradlist


if __name__ == '__main__':
    string1 = 'IPython/Jupyter搭建佳交互环境即可; 利用jupyter的cell是可以运行python文件的,即在cell中运行如下代码:"file.py"'
    string2 = '安装Python3切换 Python IPython/Jupyter搭建佳交互环境即可;在cell中运行如下代码:file.py; jupyter的cell是可以运行py文件的,'
    # plt.plot([i for i in range(len(ord_s1))], np.gradient(ord_s1))
    # plt.plot([i for i in range(len(ord_s2))], np.gradient(ord_s2))
    # plt.show()
    a = [string1, string2, string1, string2]
    b = ['''新京报讯（记者 苏季）1月21日下午，广东省教育厅发布《关于做好学校新型冠状病毒感染的肺炎疫情防控工作的紧急通知》（以下简称《通知》），要求各地各校尽量减少留校师生大型聚集性活动，要引导师生假期尽量不要前往武汉，非去不可的要做好预防措施。

适时启动教育系统应急预案和机制

近日，国家卫生健康委通报了新型冠状病毒感染的肺炎疫情，目前疫情呈现输入性、社区化、聚集性趋势，形势严峻。广东省也已出现确诊病例。为有效预防、及时控制和消除新型冠状病毒感染的肺炎疫情的危害，广东省教育厅对肺炎疫情防控工作作出安排。

《通知》要求，各地各校要充分认识疫情严峻性和复杂性，高度重视新型冠状病毒感染的肺炎等传染病防控工作。按照属地管理原则，密切关注疫情发展变化，研究部署落实防控措施，引导师生科学理性认识，适时启动教育系统公共卫生类突发事件应急预案和机制。并尽快将相关疫情防控知识和关键信息通过短信、微信、校园网等发送给师生及家长。

学校应布置寒假体育作业，鼓励师生加强锻炼，增强体质。要引导学生和家长居家或外出时做好防控工作，尽量减少到通风不畅和人流密集场所活动。要引导师生假期尽量不要前往武汉，非去不可的要做好预防措施。

要求假期到过武汉师生加强健康监测

《通知》还要求做好春节和寒假期间值班值守。各地各校结合教育实际和学校特点，摸清寒假期间在校学习生活的学生情况，通过走访慰问等方式开展面对面宣传教育，做好留校学生防控新型冠状病毒感染的肺炎等传染病工作，尽量减少留校师生大型聚集性活动。

了解和掌握师生假期动向，假期到过武汉的师生返回广东入学时，要加强健康监测。如发现学校师生员工疑似感染新型冠状病毒的肺炎疫情，要及时逐级报告至省教育厅。

另外，学校要梳理和完善传染病防控相关制度，尤其是师生和职工的晨检、因病缺勤登记与报告、隔离和复课制度等，完善学校突发公共卫生事件应急预案。高校加强发热门诊的值班力量，一旦发现传染病或疑似传染病病人，应及时报告、妥善处置。

新京报记者 苏季 校对 李铭''',string1,string2,'''新华社广州1月21日电（记者郑天虹）广东省教育厅21日发出关于做好学校新型冠状病毒感染的肺炎疫情防控工作的紧急通知，要求师生寒假期间尽量不要前往武汉，减少到通风不畅和人流密集场所活动；学校应布置寒假体育作业，加强学生体质。

　　近日，国家卫生健康委通报了新型冠状病毒感染的肺炎疫情，目前疫情呈现输入性、社区化、聚集性趋势，形势严峻。广东也已出现确诊病例。为有效预防、及时控制和消除新型冠状病毒感染的肺炎疫情的危害，保障师生员工的身体健康与生命安全，广东21日紧急发布寒假期间做好新型冠状病毒感染的肺炎疫情防控工作通知。

　　广东省教育厅要求尽快将相关疫情防控知识和关键信息通过短信、微信、校园网等发送给师生及家长。学校应布置寒假体育作业，鼓励师生加强锻炼，增强体质。要引导学生和家长居家或外出时做好防控工作，尽量减少到通风不畅和人流密集场所活动，如有不适，及时就诊。要引导师生假期尽量不要前往武汉，非去不可的要做好预防措施。

　　广东省教育厅要求各地级以上市教育局，各普通高校、省属中职学校等要充分认识疫情严峻性和复杂性，高度重视新型冠状病毒感染的肺炎等传染病防控工作，绝不能存在侥幸心理，并按照属地管理原则，适时启动教育系统公共卫生类突发事件应急预案和机制。

　　同时，做好春节和寒假期间值班值守。各地各校结合教育实际和学校特点，摸清寒假期间在校学习生活的学生情况，通过走访慰问等方式开展面对面宣传教育，做好留校学生防控新型冠状病毒感染的肺炎等传染病工作，尽量减少留校师生大型聚集性活动。了解和掌握师生假期动向，假期到过武汉的师生返回广东入学时，要加强健康监测。如发现学校师生员工疑似感染新型冠状病毒的肺炎疫情，要及时逐级报告至省教育厅。在学生开学前，做好一切环境清理和防控准备工作。''']

    o_path = getcwd().split('src')[0]+'src/'
    sys.path.append(o_path)
    from class_data_preprocessing_module import class_clearning_sting
    print(gradient_SABS_str([class_clearning_sting.delete_special_symbol(i) for i in b]))


    # 结果为：中国


