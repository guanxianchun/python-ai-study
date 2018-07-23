#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月23日
K邻近算法：采用测量不同特征值之间的距离方法进行分类
  
@author: Administrator
'''
import numpy as np
import operator

def init_datas():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify_0(in_x, train_datas, labels, k):
    """
    K邻近算法
    @param in_x: 用于分类的向量
    @param train_datas: 训练样本
    @param labels: 标签向量
    @param k: 用于选择最近邻居的数目
    """
    train_data_size = train_datas.shape[0]
    # 将输入向量扩展与训练的样本大小一致
    exten_in_x = np.tile(in_x, (train_data_size, 1))
    # 计算二点之间的距离
    distances = ((exten_in_x - train_datas) ** 2).sum(axis=1) ** 0.5
    sorted_distances_index = distances.argsort()
    class_count = {}
    # 计算与输入分类向量距离最近的k个样本的分类标签的数目
    for i in range(k):
        votal_label = labels[sorted_distances_index[i]]
        class_count[votal_label] = class_count.get(votal_label, 0) + 1
    # 对分类值进行降序排序
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 取最大值的分类标签
    return sorted_class_count[0][0]

if __name__ == '__main__':
    train_data, labels = init_datas()
    print classify_0([0.2, 0.2], train_data , labels, 3)
