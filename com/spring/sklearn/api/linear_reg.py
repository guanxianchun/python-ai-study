#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月2日

@author: Administrator
'''
from sklearn import linear_model, datasets
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
def linear_regression_example():
    # 加载糖尿病数据样本
    diabetes = datasets.load_diabetes()
    # 只使用一个特征
    diabetes_x = diabetes.data[:, np.newaxis, 2]
    # 将样本数据分为训练和测试数据(后20个为测试数据)
    diabetes_x_train = diabetes_x[:-20]
    diabetes_x_test = diabetes_x[-20:]
    # 将样本数据的Y值也分为训练和测试数据(后20个为测试数据)
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    # 创建一个线性回归对象
    linear_regression = linear_model.LinearRegression()
    # 求解线性归回参数
    linear_regression.fit(diabetes_x_train, diabetes_y_train)
    # 通过测试值预测
    diabetes_y_pred = linear_regression.predict(diabetes_x_test)
    print(u"求解的参数值 :")
    print(linear_regression.coef_)
    # 均方误差(均方差):反映估计量与被估计量之间差异程度的一种度量
    print("均方差 : %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print('R2的值 : %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    # Plot outputs
    plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
    plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.xlabel(u"测试样本值")
    plt.ylabel(u"测试Y值或估算值")
    plt.show()
if __name__ == '__main__':
   linear_regression_example()
