import numpy as np
from matplotlib import pyplot as plt


def initdataset():
    table = []
    with open("testSet.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            table.append([float(i) for i in line.split()])
    table = np.array(table)
    X = []
    Y = []
    for i in range(0,table.shape[0]):
        X1 = table[i][0:2]
        X2 = np.ones(1)
        X.append(np.concatenate([X1,X2],axis=0).reshape([3,1]))
        Y.append(table[i][2])
    trainX = X[0:80]
    trainY = Y[0:80]
    valX = X[80:100]
    valY = Y[80:100]
    return trainX,trainY,valX,valY

def tanh(vector):
    return(np.exp(vector) - np.exp(-vector)) / (np.exp(vector) + np.exp(-vector))

def sigmoid(vector):
    return 1/(1+np.exp((-1)*vector))

class net():
    layer1 = layer2 = layer3 = None
    output1 = output2 = output3 = None
    layer = [5,5]
    def __init__(self):
        self.layer1 = np.random.randn(3,self.layer[0])-0.5
        self.layer2 = np.random.randn(self.layer[0],self.layer[1])-0.5
        self.layer3 = np.random.randn(self.layer[1],1)-0.5

    def forward(self,input):
        self.output1 = sigmoid(np.dot(self.layer1.transpose(),input))
        self.output2 = sigmoid(np.dot(self.layer2.transpose(),self.output1))
        self.output3 = sigmoid(np.dot(self.layer3.transpose(),self.output2))
        return self.output3

    def backward(self,input,label,lr):
        partialA3 = np.sum(self.output3 - label,axis=1).reshape([1,1])
        #partialA3 = 1x1
        partialW3 = np.sum(self.output2 * partialA3,axis=1).reshape([5,1])
        #partialW3 = 5x1; 即每个权重分到一个梯度
        self.layer3 = self.layer3 - (partialW3 * lr).reshape([5,1])
        partialA2 = np.dot(partialA3,self.layer3.transpose())  * np.sum((self.output2*(1-self.output2)),axis=1)
        #partialA2 = 1x5; 即每个神经元分到一个梯度
        partialW2 = np.dot(np.sum(self.output1,axis=1).reshape([self.layer[0],1]),partialA2 )
        #partialW2 = 5x5; 即每个神经元的每个权重分到一个梯度
        self.layer2 = self.layer2 - (partialW2 * lr)
        partialA1 = np.dot(partialA2,self.layer2.transpose())  * np.sum((self.output1*(1-self.output1)),axis=1)
        #partialA1 = 1x5; 即每个神经元分到一个梯度
        partialW1 = np.dot( np.sum(input,axis=1).reshape([3,1]),partialA1)
        #partialW1 = 3x5; 即每个神经元的每个权重分到一个梯度
        self.layer1 = self.layer1 - (partialW1 * lr)




def val(valX,valY,net,epoch):
    a11 = a12 = a21 = loss = 0
    for i in range(len(valX)):
        Yhat = net.forward(valX[i])
        predictY = Yhat > 0.5
        loss += float(-(valY[i] * np.log(Yhat) + (1 - valY[i]) * np.log(1 - Yhat)))
        a11 += float(predictY == valY[i])
        a12 += float((predictY == 1) * predictY != valY[i])
        a21 += float((valY[i] == 1) * valY[i] != predictY)
    precision = a11 / (a11 + a12)
    recall = a11 / (a11 + a21)
    F1 = 2 * precision * recall / (precision + recall)
    print("epoch:" + str(epoch) + " precision:" + str(precision) + " recall:" + str(recall) + " F1:" + str(
        F1) + " loss:" + str(loss))

    return precision, recall, F1, loss


def train(trainX,trainY,valX,valY,net,lr,epoch):
    precision = []
    recall = []
    F1 = []
    loss = []
    for i in range(0,epoch):
        for j in range(0,len(trainX)):
            net.forward(trainX[j])
            net.backward(trainX[j],trainY[j],lr)
        if (i % 5 == 0):
            p,r,f,l= val(valX, valY, net, i)
            precision.append(p)
            recall.append(r)
            F1.append(f)
            loss.append(l)
    ep = np.linspace(5, epoch, num=int(epoch / 5))
    plt.plot(ep, precision, ls='-', lw=2, label='precision', color='purple')
    plt.plot(ep, recall, ls='-', lw=2, label='recall', color='blue')
    plt.plot(ep, F1, ls='-', lw=2, label='F1', color='green')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('value')
    plt.show()

    plt.plot(ep, loss, ls='-', lw=2, label='loss', color='purple')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.show()





trainX,trainY,valX,valY = initdataset()
lr = 0.001
epoch = 500
ljsnet = net()
train(trainX,trainY,valX,valY,ljsnet,lr,epoch)











