#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月24日

@author: Administrator
信息Xi定义：l(xi)=-log(P(xi),2)    P(xi)表示选择该分类的概率
熵定义为信息的期望值  ： H(Y|X)=-sum(P(xi) * log(P(xi),2))  i=1...n
信息增益 ： G(Y|X) = H(Y) - H(Y|X)
'''
import math
import operator

class DecisionnTree(object):
    
    def __init__(self):
        pass
    
    def create_datas(self):
        data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return data, labels
    
    def split_data(self, data, axis, value):
        """
                按指定的特征值划分数据集
        """
        ret_data = []
        for item in data:
            if item[axis] == value:
                split_item = item[:axis]
                split_item.extend(item[axis + 1:])
                ret_data.append(split_item)
        return ret_data
    
    def calc_feature_entropy_value(self, data):
        """
                计算给定数据集的香农熵
        """
        classify_count = {}
        num_entries = len(data)
        for entry in data:
            classify_label = entry[-1]
            if classify_label not in classify_count: classify_count[classify_label] = 0
            classify_count[classify_label] += 1
        entropy_vlaue = 0.0
        for key in classify_count:
            prob = float(classify_count.get(key)) / num_entries
            entropy_vlaue -= prob * math.log(prob, 2)
        return entropy_vlaue
    
    def choose_best_feature_to_split(self, data, labels):
        """
                返回样本中信息增益最大的特征
        """
        # 得到样本的特征数
        num_features = len(data[0]) - 1
        base_entropy_value = self.calc_feature_entropy_value(data)
        # 计算每一个特征的熵值
        best_info_gain = 0.0
        best_feature = -1
        # 从所有特征中找出信息增益最大的特征
        for i in range(num_features):
            # 得到i特征的所有值
            feature_values = [example[i] for example in data]
            unique_values = set(feature_values)
            new_entropy = 0.0
            # 对每一个特征值求信息增益，选取信息增益最大的特征
            for value in unique_values:
                sub_data = self.split_data(data, i, value)
                # 计算特征值的概率
                prob = len(sub_data) / float(len(data))
                # 计算特征value的熵值
                new_entropy += prob * self.calc_feature_entropy_value(sub_data)
            # 得到特征i对应的信息增益
            info_gain = base_entropy_value - new_entropy
            print "feature :%s\tbase entropy: %s\tfeature entropy: %s\tgain :%s" % (labels[i], base_entropy_value, new_entropy, info_gain)
            # 求信息增益大的特征
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature, labels[best_feature]
    
    def majority_cnt(self, classifies):
        """
        从分类标签中选出次数最多的标签分类
        """
        class_cnt = {}
        for vote in classifies:
            if vote not in class_cnt:class_cnt[vote] = 0
            class_cnt[vote] += 1
        sorted_classifies = sorted(class_cnt.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_classifies[0][0]
        
    def create_tree(self, data, labels):
        """
                构造决策树
        """
        # 得到样本中所有的分类
        classifies = [example[-1] for example in data]
        # 判断分类集合中是否只含一种分类，如果只有一种分类，则返回，不再向下分类
        if classifies.count(classifies[0]) == len(classifies):return classifies[0]
        # 只剩下最后一个特征，则返回次数最多的分类
        if len(data[0]) == 1: return self.majority_cnt(classifies)
        # 找到最好的分类特征信息
        best_features = self.choose_best_feature_to_split(data, labels)
        my_tree = {best_features[1]:{}}
        del labels[best_features[0]]
        feature_value = [example[best_features[0]] for example in data]
        unique_values = set(feature_value)
        # 对特征值进行迭代，生成相应的子树
        for value in unique_values:
            # 复制所有列名称
            sub_labels = labels[:]
            sub_data = self.split_data(data, best_features[0], value)
            sub_tree = self.create_tree(sub_data, sub_labels)
            if isinstance(sub_tree, dict):
                sub_tree["samples"] = len(sub_data)
                my_tree[best_features[1]][value] = sub_tree
            else:
                my_tree[best_features[1]][value] = {"label":sub_tree, "samples":len(sub_data)}
        return my_tree
    
if __name__ == '__main__':
    decision_tree = DecisionnTree()
    data , labels = decision_tree.create_datas()
    print "best feature index :%s\tfeature name:%s" % decision_tree.choose_best_feature_to_split(data, labels)
    print decision_tree.create_tree(data, labels)
