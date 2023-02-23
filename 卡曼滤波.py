#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   卡曼滤波.py
@Time    :   2020/08/29 15:31:55
@Author  :   manmanzhang 
@Version :   1.0
@Contact :   408903228@qq.com
@Desc    :   None
'''

# here put the import lib

import math
import numpy as np
import matplotlib.pyplot as plt

def gaussian(mu,sigma,x):
    coefficient = 1.0/np.sqrt(2.0*pi*sigma)
    exponential = np.exp(-0.5 * (x-mu) ** 2 /sigma)
    return coefficient * exponential