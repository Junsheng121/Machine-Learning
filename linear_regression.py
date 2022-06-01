import numpy as np
import matplotlib.pyplot as plt
#y=2x+1
from numpy.random import rand

def initdataset():
    #随机数据集y=2x+1
    tempk=rand(100,1)*0.1+2
    tempb=rand(100,1)*0.1+2
    tempW=np.concatenate([tempk,tempb],1)
    X1=rand(100,1)
    X2=np.ones((100,1))
    X=np.concatenate([X1,X2],1)
    Y=np.sum(tempW*X,axis=1)
    Y=Y.reshape([Y.shape[0],1])
    trainX=X[0:80,:].transpose()
    trainY=Y[0:80,:].transpose()
    valX=X[80:100,:].transpose()
    valY=Y[80:100,:].transpose()
    return trainX,trainY,valX,valY

def train(trainX,trainY,valX,valY,W,lr,epoch):
    losses = []
    for i in range(0,epoch):
        predictY = np.dot(W.transpose(),trainX)
        partialW = (1/trainX.shape[0]) * np.sum(trainX*(predictY-trainY),1).reshape([2,1])    #广播
        W = W - partialW * lr
        if(i%5==0):
            losses.append(val(valX,valY,W,i))
    ep = np.linspace(5,epoch,num=int(epoch/5))
    plt.plot(ep, losses, ls='-', lw=2, label='loss', color='purple')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.show()

def val(valX,valY,W,epoch):
    predictY = np.dot(W.transpose(),valX)
    loss = (1 / valX.shape[0]) * np.sum(pow(predictY - valY, 2),1)
    print("epoch:"+str(epoch)+" loss="+str(loss))
    print(W)
    return loss



trainX,trainY,valX,valY=initdataset()
lr = 0.0001
epoch=5000
W = np.array([[1,1]]).transpose()
train(trainX,trainY,valX,valY,W,lr,epoch)
