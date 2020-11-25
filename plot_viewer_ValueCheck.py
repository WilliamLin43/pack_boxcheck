# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # 動圖的核心函式
import seaborn as sns  # 美化圖形的一個繪圖包
from numpy import genfromtxt

sns.set_style("whitegrid")  # 設定圖形主圖

# 建立畫布
fig, ax = plt.subplots()
fig.set_tight_layout(True)

cValue = ['r','y','g','b','r','y','g','b','r'] 



def update(i):
    plt.cla()
    label = 'timestep {0}'.format(i)
    #read points record from file
    dataPath=r'./Errorlog.csv'
    data=np.asarray(genfromtxt(dataPath,delimiter=',').astype(int).tolist())
    
    #fp2 = pd.read_csv("Errorlog.csv")#read data
    #print(data)
    
    if len(data) > 100:
        VALUE1 = np.zeros(100)
        VALUE2 = np.zeros(100)
        j=0
        for k in range(len(data)):
            if k >= (len(data)-100):
                VALUE1[j] = data[k,3]
                VALUE2[j] = data[k,5]
                j=j+1
        
    else:
        VALUE1 = np.zeros(len(data))
        VALUE2 = np.zeros(len(data))
        for k in range(len(data)):
            VALUE1[k] = data[k,3]
            VALUE2[k] = data[k,5]
    

    
    #print(VALUE1)
    #print(VALUE2)
    
        
    VALUE1newData=np.array(VALUE1).transpose() #行列互換
    VALUE2newData=np.array(VALUE2).transpose() #行列互換            
        
    # 畫出一個維持不變（不會被重畫）的散點圖和一開始的那條直線。
    VALUE1newDatax = np.arange(0, len(VALUE1newData),1)
    VALUE1newDatay = VALUE1newData
    VALUE2newDatay = VALUE2newData
    #print(x)
    #print(y)
    
    #ax.scatter(x,y,c=cValue,marker='s') 
    ax.scatter(VALUE1newDatax, VALUE1newDatay,c=cValue[0])
    ax.scatter(VALUE1newDatax, VALUE2newDatay,c=cValue[1])
    
    VALUE1newDatay_Line = np.zeros(len(VALUE1newDatax))    
    VALUE2newDatay_Line = np.zeros(len(VALUE1newDatax))
    
    for j in range(len(VALUE1newDatax)):
        VALUE1newDatay_Line[j] = np.mean(VALUE1newDatay)
        VALUE2newDatay_Line[j] = np.mean(VALUE2newDatay)
    
    line, = ax.plot(VALUE1newDatax, VALUE1newDatay_Line, c=cValue[0], linewidth=2)
    line, = ax.plot(VALUE1newDatax, VALUE2newDatay_Line, c=cValue[1], linewidth=2)
    #print(label)
    # 更新直線和x軸（用一個新的x軸的標籤）。
    # 用元組（Tuple）的形式返回在這一幀要被重新繪圖的物體
    #line.set_ydata(x - 5 + i)  # 這裡是重點，更新y軸的資料
    ax.set_xlabel(label)    # 這裡是重點，更新x軸的標籤
    return line, ax

# FuncAnimation 會在每一幀都呼叫“update” 函式。
# 在這裡設定一個10幀的動畫，每幀之間間隔200毫秒
anim = FuncAnimation(fig, update, frames=np.arange(0, 5), interval=200)
plt.show()