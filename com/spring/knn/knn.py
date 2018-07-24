#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月23日
K邻近算法：采用测量不同特征值之间的距离方法进行分类
  
@author: Administrator
'''
import numpy as np
import operator
import matplotlib.pyplot as plt

dating_label_map = {"didntLike":1, "smallDoses":2, "largeDoses":3}
label_dating_map = {1:"didntLike", 2:"smallDoses", 3:"largeDoses"}
def init_datas():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def get_dating_data():
    with open("dating_data.txt") as f:
        all_lines = f.readlines()
        line_size = len(all_lines)
        return_matrix = np.zeros((line_size, 3))
        index = 0
        class_labels = []
        for line in all_lines:
            datas = line.strip().split("\t")
            return_matrix[index, :] = datas[0:3]
            class_labels.append(dating_label_map[datas[-1]])
            index += 1
        return return_matrix, class_labels

def show_picture(x_data, y_data, labels):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, 15.0 * np.array(labels), 15.0 * np.array(labels))
    plt.show()
    
    
def autoNorm(data):
    min_value = data.min(0)
    max_value = data.max(0)
    rangs = max_value - min_value
    print min_value, max_value
    m = data.shape[0]
    normal_data = data - np.tile(min_value, (m, 1))
    normal_data = normal_data / np.tile(rangs, (m, 1))
    return normal_data, rangs, min_value

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
    return label_dating_map[sorted_class_count[0][0]]

if __name__ == '__main__':
    train_data, labels = get_dating_data()  # init_datas()
    raw_data = [500, 8.9, 1.5]
    show_picture(train_data[:, 0], train_data[:, 1], labels)
#     show_picture(train_data[:, 0], train_data[:, 2], labels)
#     show_picture(train_data[:, 1], train_data[:, 2], labels)
    print 'classify: ', classify_0(raw_data, train_data, labels, 500)
    # 对原始数据进行归一化处理
    new_data, rangs, min_value = autoNorm(train_data)
    show_picture(new_data[:, 0], new_data[:, 1], labels)
    # 对输入数据进行归一化处理
    normal_data = (np.array(raw_data) - min_value) / rangs
    print 'classify: ', classify_0(normal_data, new_data, labels, 500)
