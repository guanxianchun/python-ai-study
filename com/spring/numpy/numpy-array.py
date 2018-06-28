#/usr/bin/env python
#-*- encoding:utf8 -*-
'''
Created on 2018年6月21日

@author: root
'''
import numpy as np
def numpy_init():
    #init array data 
    array_data=np.array([1,2,3,4,5,6])
    print(array_data)
    #init zero data  3 row 4 colum
    zero_data = np.zeros((3,4))
    print(zero_data)
    #init one data  3 row 4 colum
    one_data = np.ones((3,4))
    print(one_data)
    
def numpy_equal():
    array_data=np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(">>> show data :")
    print(array_data)
    print(">>> 获取第2行的数据 :")
    print(array_data[1])
    print(">>> 获取第2列的数据 :")
    print(array_data[:,1])
    print(">>> 获取等于6的数据:")
    print(array_data==6)
    print(array_data[array_data==6])
    print(">>> 获取模2等于0的数据:")
    print(array_data%2==0)
    print(array_data[array_data%2==0])
    print(">>>获取第2列数据等于5的行数据:")
    equal_value = (array_data[:,1]==5)
    print(array_data[equal_value,:])
    
def modify_value():
    array_data=np.array([[1,2,3],[4,5,6],[7,8,9]])
    equal_value = (array_data==3) | (array_data==2)
    array_data[equal_value]=10
    print(">>>将矩阵中数等于2或3修改为10")
    print(array_data)
def convert_data_type():
    array_data=np.array(['1','2','3'])
    print(array_data.astype(float))
    
def array_sum():
    array_data=np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(">>>按列求和 :")
    print(array_data.sum(axis=0))
    print(">>>按行求和:")
    print(array_data.sum(axis=1))
    
def array_reshape():
    array_data=np.array([1,2,3,4,5,6,7,8,9])
    print(">>>将向量变成矩阵:")
    print(array_data.reshape(3,3))
    
def matrix_operator():
    array_data=np.array(np.random.random(12))
    matrix_A = np.array([10,20,30,10]).reshape(4,-1)
    print(array_data)
    matrix=array_data.reshape(3,-1)
    print(">>>将向量变成矩阵:")
    print(matrix)
    print(matrix_A)
    print(">>>矩阵的乘:")
    print(matrix.dot(matrix_A))
    print(">>>矩阵的逆矩阵: ")
    print(matrix.T)
if __name__ == '__main__':
    numpy_init()
    print("*"*80)
    numpy_equal()
    print("*"*80)
    modify_value()
    print("*"*80)
    convert_data_type()
    array_sum()
    array_reshape()
    matrix_operator()
