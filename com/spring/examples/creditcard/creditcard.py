#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月2日
信用卡欺诈
@author: Administrator
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import linear_model, cross_validation
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import itertools
def get_orign_data():
    data = pd.read_csv("creditcard.csv")
    return data


def view_orign_data(data):
    count_classes = pd.value_counts(data["Class"], sort=True).sort_index()
    count_classes.plot(kind='bar')
    plt.title(u"欺诈类直方图")
    plt.xlabel(u"欺诈类")
    plt.ylabel(u"欺诈频率")
    plt.show()

def plot_confusion_matrix(conf_matrix, classes, title=u"混淆矩阵", cmap=plt.cm.Blues):  
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def down_sample_cross_validate(data):
    """
    下采样:使二分类样本中,如果目标值为1的样本很少,目标值为0的样本很多,我们从目标值为0的样本中取出与目标值为1一样多的样本
                然后将它们组合在一起.使二个样本数据同样少
    """
    # 得到样本数据和目标值
    x = data.iloc[:, data.columns != "Class"]
    y = data.iloc[:, data.columns == "Class"]
    # 得到欺诈的样本数据有多少
    number_records_fraud = len(data[data["Class"] == 1])
    fraud_indices = np.array(data[data["Class"] == 1].index)
    normal_indices = data[data["Class"] == 0].index
    # 随机从正常样本中取出number_records_fraud个样本
    random_normal_indices = np.array(np.random.choice(normal_indices, number_records_fraud, replace=False))
    # 将二个样本添加在一起
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    # 得到下采样的数据集
    under_sample_data = data.iloc[under_sample_indices, :]
    x_under_sample = under_sample_data.iloc[:, under_sample_data.columns != "Class"]
    y_under_sample = under_sample_data.iloc[:, under_sample_data.columns == "Class"]
    x_train, x_test, y_train, y_test = cross_validate(x, y)
    x_train_under_sample, x_test_under_sample, y_train_under_sample, y_test_under_sample = cross_validate(x_under_sample, y_under_sample)
    best_c_param = sample_kfold_scores(x_train_under_sample, y_train_under_sample)
    train_predict(best_c_param, x_train_under_sample, y_train_under_sample, x_test_under_sample, y_test_under_sample)
    train_predict(best_c_param, x_train_under_sample, y_train_under_sample, x_test, y_test)
    # 用全部测试样本去训练,预测值
    best_c_param = sample_kfold_scores(x_train, y_train)
    train_predict(best_c_param, x_train, y_train, x_test, y_test)

    
def train_predict(best_c_param, x_train, y_train, x_predict, y_test, threshold=0.6):
    """
    通过训练样本得到模型,然后用预测样本预测,再计算recall的值
    @param best_c_param: 
    @param x_train: 
    @param y_train: 
    @param x_predict: 
    @param y_test: 
    @param threshold: 预值,预测的概率值大于预值时,我们才认为是1
    """
    lr = linear_model.LogisticRegression(C=best_c_param, penalty="l2")
    lr.fit(x_train, y_train.values.ravel())
    if threshold is None:
        y_pred = lr.predict(x_predict.values)
    else:
        y_pred = lr.predict_proba(x_predict)
        y_pred = y_pred[:, 1] > threshold
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 计算recall值 = TP/(TP+FN)
    # 真实值
    #    |
    #  0 |  TN(负类->负类)    FP(负类->正类)
    #    |
    #  1 |  FN(正类->负类)    TP(正类->正类)
    #    |________________________________________
    #            0                 1           预测值
    print("召回率:反映被正确判断为正例占总的正例的比重,公式: TP/(TP+FN)")
    recall_rate = conf_matrix[1, 1] * 1. / (conf_matrix[1, 1] + conf_matrix[1, 0])
    print("测试样本中召回率 R : %s" % (recall_rate))
    print("精度: 反映被分类器判定的正例中,真正的正例所占比重,公式: TP/(TP+FP)")
    pression = conf_matrix[1, 1] * 1. / (conf_matrix[1, 1] + conf_matrix[0, 1])
    print("精度 P : %s" % (pression))
    print("F1 Score (2*P*R/(P+R) : %s" % (2 * pression * recall_rate / (pression + recall_rate)))
    print("转移性: 负例判断为负例占总的负例样本的比重, 公式: TN/(TN+FP)")
    print("转移性S: %s" % (conf_matrix[0, 1] * 1. / (conf_matrix[0, 0] + conf_matrix[0, 1])))
    plot_confusion_matrix(conf_matrix, classes=[0, 1])
    
def sample_kfold_scores(x_train_data, y_train_data):
    # 将训练数据分成多少等份,这里是分成5等份
    flod = cross_validation.KFold(len(y_train_data), 5, shuffle=False)
    # 设置惩罚力度
    init_data = np.array([0.01, 0, 0.1, 0, 1, 0, 10, 0, 100, 0]).reshape(-1, 2)
    result_table = pd.DataFrame(init_data, columns=["C_Params", "Mean recall score"])
    j = 0
    for c_param in result_table["C_Params"]:
        recall_accs = []
        print("C parameter : %s" % c_param)
        for iter, indices in enumerate(flod, start=1):
            # 创建逻辑回归对象,传入二个参数 C:惩罚值   penalty:惩罚类型(L1:|w|,L2:1/2*w**2)
            lr = linear_model.LogisticRegression(C=c_param, penalty="l2")
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            y_pred_under_sample = lr.predict(x_train_data.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :], y_pred_under_sample)
            recall_accs.append(recall_acc)
            print("Iteration  %s : recall score = %s" % (iter, recall_acc))
        result_table.loc[j, "Mean recall score"] = np.mean(recall_accs)
        j += 1
        print("Mean recall score : %s" % (np.mean(recall_accs)))
    
    best_c_param = result_table.loc[result_table["Mean recall score"].idxmax()]["C_Params"]
    print("*"*80)
    print(result_table)
    print("*"*80)
    print("Best model to choose from cross validation is with C parameter = %s" % (best_c_param))
    print("*"*80)
    return best_c_param

def over_sample_cross_validate(data):
    """
        过采样:使二分类样本中,如果目标值为1的样本很少,目标值为0的样本很多,我们生成一些目标值为1的样本,使其与目标值为1一样多的样本
                然后将它们组合在一起.使二个样本数据同样多
    """
    columns = data.columns
    # 去掉最后一列(Class)
    feature_columns = columns.delete(len(columns) - 1)
    # 获取特征值和目标值
    features = data[feature_columns]
    labels = data["Class"]
    
    # 获取交叉训练和测试样本
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    # 得到过采样的训练样本
    oversampler = SMOTE(random_state=0)
    over_sample_features, over_sample_labels = oversampler.fit_sample(feature_train, label_train)
    over_features = pd.DataFrame(over_sample_features)
    over_labels = pd.DataFrame(over_sample_labels)
    best_c_param = sample_kfold_scores(over_features, over_labels)
    train_predict(best_c_param, over_features, over_labels, feature_test, label_test, 0.5)
    
def standar_scaler(data):
    """
        标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
        对数据进行标准化处理,机器学习算法可能存在这样的一个误区:
            认为特征值比较大的,重要程度比较大,特征值比较小的,重要程度比较小
            因此,我们需要对Amount列的使其在(-1,1)的区间上
    """
    data["normAmount"] = StandardScaler().fit_transform(np.array(data.loc[:, "Amount"]).reshape(-1, 1))
    data = data.drop(["Time", "Amount"], axis=1)
    return data

def cross_validate(x, y):
    """
       交叉验证
    """
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test
if __name__ == '__main__':
    orign_data = get_orign_data()
    # 查看样本数据(0表示正常的数据,1:异常的数据)
#     view_orign_data(orign_data)
    data = standar_scaler(orign_data)
#     down_sample_cross_validate(data)
    print("-------------------------过采样------------------------")
    over_sample_cross_validate(orign_data)
    
