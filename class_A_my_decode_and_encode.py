import numpy as np
import random

# 明文
# strings = '192.168.1.1 :3306 这是一个测试  must to do it'

# ：TODO 这是一个加解密的函数文件 decodeofmyself是加密函数，encodeofmyself是解密函数

# 密钥
keypoint = random.randint(1, 20000)
key = ''.join([chr(random.randint(65, i)) for i in range(65 + keypoint, 65 + 100 + keypoint)])
print(key)


def decodeofmyself(string, key):
    '''
    :param string: -> plaintext
    :param key: -> key
    :return: -> cipher
    '''
    mykey = np.array(
        [ord(a) for a in [i for i in key] * int(len(string) // len(key)) + list(key[0:len(string) % len(key)])])
    mynewstring = np.array([ord(e) for e in string])
    return ''.join([hex(a) for a in mynewstring + mykey])


def encodeofmyself(inter, key):
    '''
    :param inter: -> cipher
    :param key: -> key
    :return: -> plaintext
    '''
    mysplit = np.array([int('0x' + e, 16) for e in inter.split('0x')[1:]])
    mykey = np.array(
        [ord(a) for a in [i for i in key] * (int(len(mysplit) // len(key))) + list(key[0:len(mysplit) % len(key)])])
    return ''.join([chr(i) for i in mysplit - mykey])


if __name__ == '__main__':
    print('请输入内容：')
    strings = input()
    print('输入内容核对：', strings)
    print('密文:', '{' + decodeofmyself(strings, key) + '}')
    print('密钥:', '{' + key + '}')
    print('解密:', encodeofmyself(decodeofmyself(strings, key), key))

