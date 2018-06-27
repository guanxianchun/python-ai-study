#/usr/bin/env python
#-*- encoding:utf8 -*-
'''
Created on 2018年6月26日

@author: root
'''
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(size=(20,6))+np.arange(6)/2
def draw_sinplot(flip=1):
    """
    使用matplotlib画图
    """
    sns.set_context("notebook", font_scale=1.5,rc={"lines.linewidth":2.5})
    
    x=np.linspace(0, 14, 100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*.5)*(7-i)*flip)
    
    sns.despine(offset=10)
#     plt.figure(figsize=(8,6))
    plt.show()
    
def draw_seaborn():
    sns.set_style("ticks")   #white,whitegrid ,dark,ticks
    sns.boxplot(data=data)
    sns.despine(offset=10)
def color_demo():
#     sns.boxplot(data=np.random.normal(size=(20,10))+np.arange(10)/2,palette=sns.color_palette("hls", 10))
    sns.palplot(sns.color_palette("hls", 8))
    plt.show()
if __name__ == '__main__':
#     draw_seaborn()
    color_demo()


#     draw_sinplot(-1)
#     plt.show()
    
#     draw_sinplot()
    
    