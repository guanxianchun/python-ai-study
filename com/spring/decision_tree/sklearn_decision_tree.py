#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月27日

@author: Administrator
'''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
import numpy as np

class DecisionTree(object):
    
    def __init__(self):
        pass
    
    def get_datas(self):
        """
        获取加利福尼亚的房屋信息
        """
        return fetch_california_housing()
    
    def create_decision_tree(self):
        #得到加利諨尼亚的房屋数据
        housing = self.get_datas()
        """
                        回归决策树参数说明：
        criterion:特征选择标准：mse(均方差)和mae(和均值之差的绝对值之和)
        spliter:特征划分点标准选择：best和random,前者在特征的所有划分点中找出最优的划分点。
                                                后者是随机的在部分划分点中找局部最优的划分点。
        max_path:决策树最深度
        min_samples_split:内部节点再划分所需最小样本数，这个值限制了子树继续划分的条件，
                                            如果某节点的样本数少于该值则不会再尝试选择最优特征来进行划分。
        min_samples_leaf:叶子节点最小样本数，这个值限制了叶子节点最小的样本数，如果某叶子节点数目小于这个值，
                                                    则会和兄弟节点一起被剪枝。 默认是1，可以输入最小样本的整数，或最小样本数占样本总数的百分比。
        min_weight_fraction_leaf:叶子节点最小样本权重和，这个值限制了叶子节点所有样本权重和的最小值，
                                                如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。
        
        """
        dtr = tree.DecisionTreeRegressor(max_depth=2)
        dtr.fit(housing.data[:, [6, 7]], housing.target)
        dot_data = tree.export_graphviz(dtr, out_file=None, feature_names=housing.feature_names[6:8],
            filled=True, impurity=False, rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.get_nodes()[7].set_fillcolor("#FFF2DD")
        graph.write_png("dtr_white_background.png")
if __name__ == '__main__':
    decision_tree = DecisionTree()
    house_datas = decision_tree.get_datas()
    decision_tree.create_decision_tree()
