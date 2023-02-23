import numpy as np
import matplotlib.pyplot as plt

def case(x):
    # 概率 
    temp = 0
    for i in range(x):
        temp += np.random.randint(0,2)==1
    return temp/x

def test_group(x,y,z=0.03):
    # 置信度
    temp = 0
    for i in range(x):
        temp += (abs(0.5-case(y))<z)
    return temp/x

for i in range(10):
  print('置信度：',test_group(100,10000,0.03),"精度误差:3%")





   
