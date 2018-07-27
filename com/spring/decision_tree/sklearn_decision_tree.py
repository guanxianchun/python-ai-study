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
    
    def do_some_thing(self):
        housing = self.get_datas()
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
    decision_tree.do_some_thing()
