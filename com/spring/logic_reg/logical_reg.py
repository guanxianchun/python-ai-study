#/usr/bin/env python
#-*- encoding:utf8 -*-
'''
Created on 2018年6月28日
逻辑回归：
   目标：建立分类器（求解出三个参数，样本特征是2 如果样本的特征是n即求解的参数个数为n+1）
   设定阀值，根据阀值判断录取结果
要完成的模块：
 1.sigmoid:映射到概率的函数
 2.model:返回预测结果值
 3.cost:根据参数计算损失
 4.gradient:计算每个参数的梯度方向
 5.descent:进行参数更新
 6.accuracy:计算精度
@author: root
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def read_data():
    pdData = pd.read_csv("LogiReg_data.txt",header=None,names=["Exam1","Exam2","Admitted"])
    return pdData

def draw_picture(data):
    positive = data[data["Admitted"]==1]
    negative = data[data["Admitted"]==0]
    fig,ax = plt.subplots(figsize=(10,5))
    ax.scatter(positive["Exam1"],positive["Exam2"],s=30,c='b',marker="o",label="Admitted")
    ax.scatter(negative["Exam1"],negative["Exam2"],s=30,c='r',marker="x",label="Not Admitted")
    ax.legend()
    ax.set_xlabel("Exam 1 Score")
    ax.set_ylabel("Exam 2 Score")
    plt.show()
    
def sigmoid(z):
    """
    @param z: 样本值
    得到将样本值映射到平面的概率
    """
    return 1/(1+np.exp(-z))

def model(X,theta):
    """
    预测函数
    @param X: 
    @param theta: 
    """
    return sigmoid(np.dot(X,theta.T))

def cost(x,y,theta):
    """
    """
    left = np.multiply(-y,np.log(model(x, theta)))
    right = np.multiply(1-y,np.log(1-model(x, theta)))
    return np.sum(left-right)/len(x)

def gradient(x,y,theta):
    grad = np.zeros(theta.shape)
    error = (model(x, theta)-y).ravel()
    for j in range(len(theta.ravel())):
        item = np.multiply(error,x[:,j])
        grad[0,j] = np.sum(item)/len(x)
    return grad

STOP_ITEM=0 #迭代次数
STOP_COST=1 #损失策略
STOP_GRAD=2 #梯度策略
def stopCriterion(stopType,value,threshold):
    """
    设定三种不同的停止策略
    """
    if stopType == STOP_ITEM: return value>threshold                      #当迭代次数达到指定的次数时停止
    elif stopType== STOP_COST: return abs(value[-1]-value[-2])<threshold  #前后二次迭代之间的目标值小于threshold时停止
    elif stopType==STOP_GRAD: return np.linalg.norm(value)<threshold      #当梯度值小于指定的值时停止
    
def shuffleData(data):
    """
    洗牌，把样本数据打乱（数据可能有一定规则，为了让模型的泛化能力更强，把数据打乱)
    """
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:,0:cols-1]
    y = data[:,cols-1:cols]
    return x,y

def descent(data,theta,batchSize,stopType,thresh,alpha):
    """
    
    """
    init_time = time.time()
    i = 0  #迭代次数
    k = 0  #样本迭代的下标 
    #将样本数据打散
    x,y = shuffleData(data)
    #初始化梯度值
    grad = np.zeros(theta.shape)
    #记录样本每次迭代的损失值
    costs = [cost(x, y, theta)]
    while True:
        #计算梯度
        grad = gradient(x[k:k+batchSize], y[k:k+batchSize], theta)
        k+=batchSize
        if k>=len(data):
            k=0
            x,y = shuffleData(data)
        #更新求解的参数值theta
        theta = theta -alpha*grad
        costs.append(cost(x,y,theta))
        i+=1
        if stopType == STOP_ITEM : value =i
        elif stopType == STOP_COST: value = costs
        elif stopType == STOP_GRAD: value = grad
        #判断不同的迭代停止策略，迭代是否停止
        if stopCriterion(stopType, value, thresh):break
    return theta,i-1,costs,grad,time.time()-init_time

def runExpe(data,theta,batchSize,stopType,thresh,alpha):
    #进行梯度下降算法进行求值(theta值)
    theta,_iter,costs,_grad,dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum()>1 else "Scaled" 
    name+=" data - learning rate {} - ".format(alpha)
    if batchSize== len(data) :strDescType="Gradient"
    elif batchSize==1:strDescType="Stochastic"
    else:strDescType="Mini-batch ({})".format(batchSize)
    name+=strDescType+" descent -Stop:"
    if stopType == STOP_ITEM : strStop ="{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop ="costs change <{}".format(thresh)
    else: strStop ="gradient norm < {} ".format(thresh)
    name+=strStop
    print("***{}\nTheta: {} - Iter {} Last cost :{:03.2f} -Duration: {:03.2f}s".format(name,theta,_iter,costs[-1],dur))
    fig,ax = plt.subplots(figsize=(12,6))
    ax.plot(np.arange(len(costs)),costs,'r')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Costs")
    ax.set_title(name.upper()+" - Error vs. Iteration")
    return theta

if __name__ == '__main__':
    data = read_data()
    data.insert(0, "Ones", 1)  #增加一列它的值为1  
    orig_data = data.get_values()
    theta = np.zeros([1,3])
    batchSize=100
    #STOP_ITEM 迭代样本次数策略  n=100（每次迭代样本数）  thresh=5000(指定迭代次数)    alpha （学习率） 
#     theta = runExpe(orig_data, theta, batchSize, STOP_ITEM, thresh=5000, alpha=0.000001)
    #STOP_COST 迭代损失策略  不指定迭代数，指定二次预值之间的差距阀值thresh，当小于阀值时停止迭代 
    theta = runExpe(orig_data, theta, batchSize, STOP_COST, thresh=0.000001, alpha=0.001)
    #STOP_GRAD 迭代梯度策略  指定梯度值阀值thresh＝0.05 当迭代的梯度值小于梯度阀值时停止迭代 
#     theta = runExpe(orig_data, theta, batchSize, STOP_GRAD, thresh=0.05, alpha=0.001)
    print(theta)
    plt.show()
