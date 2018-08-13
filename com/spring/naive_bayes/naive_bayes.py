#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年8月13日
朴素贝叶斯：选择具有最高概率的决策
P(C1|X)：表示给定X，那么它来自类别C1的概率,X(X1,X2...Xn)
P(C1|X)=P(X|C1)*P(C1)/P(X)
取对数似然将乘积变成求和
logP(C1|X)=logP(X|C1)*P(C1)/P(X) =logP(X|C1)+logP(C1) -logP(X)
因为logP(X)是固定的，
所以只要比较logP(X|C1)+logP(C1)的大小即可
@author: guan.xianchun
'''
import numpy as np
import re, random
class NaiveBayes(object):
    
    def __init__(self):
        pass
    
    def load_datas(self):
        datas = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        class_vector = [0, 1, 0, 1, 0, 1]  # 1:代表侮辱性文字　0：表示正常言论
        return datas, class_vector
    
    def create_vocabulary(self, datas):
        """
                获取数据库中所有不重复的词的列表
        @param datas: 
        """
        vocabulary_set = set([])
        for document in datas:
            vocabulary_set = vocabulary_set | set(document)
        return list(vocabulary_set)
    
    def set_word_vector(self, vocabulary, inputs):
        """
                记录词汇表中的词是否出现在输入的文档中
                文档向量化，这里是词袋模型，不仅关心某个词条出现与否，还考虑该词条在本文档中的出现频率
        @param vocabulary:词汇表
        @param inputs: 输入文档的词集合 
        @return: 文档向量(长度与词汇表等长)
        """
        # 每个文档的大小与词典保持一致
        return_vector = [0] * len(vocabulary)
        for word in inputs:
            if word in vocabulary:
                # 当前文档中的某个词存在词汇表中，则将其相应的位置+1，即记录词出现的次数
                return_vector[vocabulary.index(word)] += 1  
#             else:print "the word :%s is not in my vocabulary " % (word)
        return return_vector
    def train_naive_bayes(self, train_matrix, train_category):
        """
        
        """
        num_trains = len(train_matrix)
        num_words = len(train_matrix[0])
        # 计算侮辱性文档的概率
        prob_abusive = np.sum(train_category) / float(num_trains)
        # 计算p(w0|1)p(w1|1)p(w2|1)。如果其中一个概率值为0，那么最后的乘积也为0。为降低 这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2
        prob0_num = np.ones(num_words)
        prob1_num = np.ones(num_words)
        prob0_denom = 2.0
        prob1_denom = 2.0
        for i in range(num_trains):
            if train_category[i] == 1:
                prob1_num += train_matrix[i]
                prob1_denom += np.sum(train_matrix[i])
            else:
                prob0_num += train_matrix[i]
                prob0_denom += np.sum(train_matrix[i])
        # 统计词典中所有词条在正常文档中出现的概率,为了防止下溢出问题，取对数  logA*B = logA+logB  A*B值非常小时结果会为0
        prob0_vector = np.log(prob0_num / prob0_denom)
        # 统计词典中所有词条在侮辱性文档中出现的概率
        prob1_vector = np.log(prob1_num / prob1_denom)
        return  prob0_vector, prob1_vector, prob_abusive
    
    def classify_naive_bayes(self, classify_vector, prob0_vector, prob1_vector, prob_class1):
        # logP(Xi|C1)+logP(C1)  其中Xi表示第i个特征
        prob1 = np.sum(classify_vector * prob1_vector) + np.log(prob_class1)
        # logP(Xi|C0)+logP(C0)  其中Xi表示第i个特征
        prob0 = np.sum(classify_vector * prob0_vector) + np.log(1.0 - prob_class1)
        # 取概率最高的决策
        return 1 if prob1 > prob0 else 0
    
    def testing_naive_bayes(self):
        list_posts , list_classes = self.load_datas()
        my_vocabulary = self.create_vocabulary(list_posts)
        train_matrix = []
        for post_in_doc in list_posts:
            train_matrix.append(self.set_word_vector(my_vocabulary, post_in_doc))
        p0_vector, p1_vector, p_abusive = self.train_naive_bayes(np.array(train_matrix), np.array(list_classes))
        test_entry = ['love', 'my', 'dalmation']
        this_doc = np.array(self.set_word_vector(my_vocabulary, test_entry))
        print test_entry, ' classified as ', self.classify_naive_bayes(this_doc, p0_vector, p1_vector, p_abusive)
        test_entry = ['stupid', 'garbage']
        this_doc = np.array(self.set_word_vector(my_vocabulary, test_entry))
        print test_entry, ' classified as ', self.classify_naive_bayes(this_doc, p0_vector, p1_vector, p_abusive)
    
    def text_parse(self, big_string):
        
        tokens = re.split(r'\W*', big_string)
        return [tok.lower() for tok in tokens if len(tok) > 2]
    
    def set_date_from_file(self, file_name, class_value, doc_list, full_text, class_list):
        with open(file_name) as f:
            words = self.text_parse(f.read())
            doc_list.append(words)
            full_text.extend(words)
            class_list.append(class_value)
    def testing_naive_bayes_from_file(self):
        doc_list = []
        class_list = []
        full_text = []
        for i in range(1, 26):
            self.set_date_from_file('email_spam/%d.txt' % (i), 1, doc_list, full_text, class_list)
            self.set_date_from_file('email_ham/%d.txt' % (i), 0, doc_list, full_text, class_list)
        vocabulary = self.create_vocabulary(doc_list)
        # 贝叶斯交叉验证,取40个做训练集，10个做测试数据
        train_indexs = range(50)
        test_indexs = []
        for i in range(10):
            random_index = int(random.uniform(0, len(train_indexs)))
            test_indexs.append(train_indexs[random_index])
            del train_indexs[random_index]
        train_matrix = []
        train_classes = []
        for doc_index in train_indexs:
            train_matrix.append(self.set_word_vector(vocabulary, doc_list[doc_index]))
            train_classes.append(class_list[doc_index])
        p0_vector, p1_vector, p_spam = self.train_naive_bayes(np.array(train_matrix), np.array(train_classes))
        error_count = 0  # 记录测试集上预测错误的次数
        for doc_index in test_indexs:
            words = self.set_word_vector(vocabulary, doc_list[doc_index])
            if self.classify_naive_bayes(np.array(words), p0_vector, p1_vector, p_spam) != class_list[doc_index]:
                error_count += 1
                print 'classification error :', doc_list[doc_index]
        print 'The error rate is :', float(error_count) / len(test_indexs)
if __name__ == '__main__':
    naive_bayes = NaiveBayes()
    naive_bayes.testing_naive_bayes()
    naive_bayes.testing_naive_bayes_from_file()
