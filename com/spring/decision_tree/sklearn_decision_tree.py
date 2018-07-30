#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年7月27日

@author: Administrator
'''
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus, pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class DecisionTree(object):
    
    def __init__(self):
        pass
    
    def get_datas(self):
        """
        获取加利福尼亚的房屋信息
        """
        housing = fetch_california_housing()
        return housing
    
    def create_decision_tree(self):
        # 得到加利諨尼亚的房屋数据
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
        max_features:划分时考虑的最大特征数，可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；
                                            如果是"log2"意味着划分时最多考虑log2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑N−−√个特征。
                                            如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。
                                            一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
        max_leaf_nodes:通过限制最大叶子数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。
                                            如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制具体的值可以通过交叉验证得到。
        min_impurity_split:这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值则该节点不再生成子节点。即为叶子节点 。
        n_estimators:要建立树的个数
        """
        dtr = tree.DecisionTreeRegressor(max_depth=5)
        # 构造一个回归决策树
        dtr.fit(housing.data[:, [2, 3]], housing.target)
        # 生成可视化树
        self.build_tree_image(dtr, housing.feature_names[2:4], "dtr_white_background.png")
    
    def create_regession_decision_tree(self):
        """
        交叉验证生成决策树
        """
        housing = self.get_datas()
        train_data, test_data, train_target, test_target = train_test_split(housing.data, housing.target, test_size=0.1, random_state=42)
        dtr = tree.DecisionTreeRegressor(random_state=42)
        dtr.fit(train_data, train_target)
        # 返回预测的系数r ^ 2，该值越大越好
        print "回归决策树，预测的系数R**2: %s" % dtr.score(test_data, test_target)
        
    def build_tree_image(self, decission_tree, feature_names, file_name):
        dot_data = tree.export_graphviz(decission_tree, out_file=None, feature_names=feature_names,
                                        filled=True, impurity=False, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.get_nodes()[7].set_fillcolor("#FFF2DD")
        graph.write_png(file_name)
        
    def get_cross_train_test_data(self, test_size=0.2):
        housing = self.get_datas()
        train_data, test_data, train_target, test_target = train_test_split(housing.data, housing.target, test_size=test_size, random_state=42)
        return train_data, test_data, train_target, test_target, housing.feature_names
    
    def save_random_regression_module(self, rfr, file_name):
        """
                把训练好的随机森林存进pickle中了，省去了每次训练，其实并不是所有的类都能存进pickle， 
                但随机森林是可以的，这里不局限于存一个model，你可以把训练集、训练集的标签、随机森林放到一个list一起扔进pickle
        """
        with open(file_name, 'wb') as f:
            pickle.dump(rfr, f)
        
    def get_random_regression_module(self, file_name):
        """
                从pickle中读取训练好的数据模型
        """
        with open(file_name) as f :
            return pickle.load(f)
        
    def create_random_forest_tree(self):
        """
                随机森林生成决策树
        """
        def gride_search_cv(train_data, train_target):
            tree_param_grid = { 'min_samples_split': list((3, 6, 9)), 'n_estimators':list((10, 50, 100))}
            grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5, n_jobs=4)
            grid.fit(train_data, train_target)
            print grid.grid_scores_
            print grid.best_params_
            print grid.best_score_
            
        train_data, test_data, train_target, test_target, feature_names = self.get_cross_train_test_data(0.4)
#         gride_search_cv(train_data, train_target)
        # 根据上面得到最优的参数，重新生成决策树，用测试集进行评估
        #
        rfr = RandomForestRegressor(min_samples_split=30, n_estimators=10, random_state=42)
        rfr.fit(train_data, train_target)
        print "随机森林回归，预测的系数R**2: %s" % rfr.score(test_data, test_target)
#         self.save_random_regression_module(rfr, "random_forest_reg_tree.pkl")
        self.build_tree_image(rfr.estimators_[0], feature_names, "random_forest_reg_tree.png")
if __name__ == '__main__':
    decision_tree = DecisionTree()
    house_datas = decision_tree.get_datas()
#     decision_tree.create_decision_tree()
#     decision_tree.create_regession_decision_tree()
    decision_tree.create_random_forest_tree()
