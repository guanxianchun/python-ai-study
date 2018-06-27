#/usr/bin/env python
#-*- encoding:utf8 -*-
'''
Created on 2018年6月26日

@author: root
'''
import matplotlib.pyplot as plt
import numpy as np
def draw_line_piture():
    plt.plot(["2018-01","2018-02","2018-03","2018-04","2018-05","2018-06","2018-07","2018-08"],[20,30,40,30,60,20,30,80],c='red',label='销售团队A')
    plt.plot(["2018-01","2018-02","2018-03","2018-04","2018-05","2018-06","2018-07","2018-08"],[30,40,10,20,50,30,50,85],c='b',label='销售团队B')
    plt.xticks(rotation=45) #让x轴的标签字旋转45度
    plt.xlabel("年月")
    plt.ylabel("销售额(万)")
    plt.title("公司每月的销售情况")
    #'best','upper right','upper left','lower left','lower right','right',
    #'center left','center right','lower center','upper center','center'
    plt.legend(loc='upper center')  
#     print(help(plt.legend)) #查看legend的用法
    plt.show()
def draw_sub_piture():
    fig = plt.figure()
    for i in range(3):
        axi = fig.add_subplot(2,2,i+1)
        axi.set_label("dddddddd")
        print(help(axi))
    plt.show()
    
def draw_bar_piture():
    fig,ax = plt.subplots()
    x_label=["2018-01","2018-02","2018-03","2018-04","2018-05","2018-06","2018-07","2018-08"]
    data=[20,30,40,30,60,20,30,80]
    ax.bar(x_label,data,0.5)
    ax.set_xticks(range(0,8))
    ax.set_xticklabels(x_label,rotation=45)
    ax.set_xlabel("年月")
    ax.set_ylabel("销售额(万)")
    ax.set_title("公司每月的销售情况")
    print(help(ax.bar))
    plt.show()
    
def draw_scatter_picture():
    fig,ax = plt.subplots()
    x_label=[]
    data=[20,30,40,30,60,20,30,80]
    ax.scatter(x_label,data,0.5)
    ax.set_xticks(range(0,8))
    ax.set_xticklabels(x_label,rotation=45)
    ax.set_xlabel("年月")
    ax.set_ylabel("销售额(万)")
    ax.set_title("公司每月的销售情况")
    plt.show()
    
if __name__ == '__main__':
#     draw_line_piture()
#     draw_sub_piture()
#     draw_bar_piture()
    draw_scatter_picture()