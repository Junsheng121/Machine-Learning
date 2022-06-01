import numpy as np
from matplotlib import pyplot as plt
def sigmoid(vector):
    return 1/(1+np.exp((-1)*vector))

def Gaussian_distance(x,core,sigma):
    result = []
    for i in range(x.shape[1]):
        vec = core - x[:,i].reshape([3,1])
        vec2 = np.dot(vec.transpose(), vec)
        vec3 = []
        for j in range(core.shape[1]):
            vec3.append(vec2[j,j])
        vec2 = np.array(vec3)
        result.append(np.concatenate([np.exp(-1*vec2/(sigma**2)),[1]]))
    return np.array(result).transpose()

def initdataset():
    table = []
    with open("testSet.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            table.append([float(i) for i in line.split()])
    table = np.array(table)
    X1 = table[:,0:table.shape[1]-1]
    X2 = np.ones((table.shape[0],1))
    X = np.concatenate([X1,X2],axis=1)
    Y = table[:,table.shape[1]-1:]
    trainX = X[0:int(0.8 * X.shape[0]),:].transpose()
    trainY = Y[0:int(0.8 * Y.shape[0]),:].transpose()
    valX = X[0:int(0.2*X.shape[0]),:].transpose()
    valY = Y[0:int(0.2 * Y.shape[0]),:].transpose()
    return trainX,trainY,valX,valY
def val(valX,valY,W,epoch):
    Yhat=sigmoid(np.dot(W.transpose(),valX))
    predictY = Yhat>0.5
    loss = -np.sum(valY*np.log(Yhat)+(1-valY)*np.log(1-Yhat),1)
    a11=np.sum(predictY==valY,1)
    a12=np.sum((predictY==1) * predictY!=valY,1)
    a21=np.sum((valY==1)*valY!=predictY,1)

    precision = a11/(a11+a12)
    recall = a11/(a11+a21)
    F1 = 2*precision*recall/(precision+recall)
    print("epoch:"+str(epoch)+" precision:"+str(precision)+" recall:"+str(recall)+" F1:"+str(F1)+" loss:"+str(loss))
    return precision,recall,F1,loss
    #recall = np.sum(predictY==)
def train(trainX,trainY,valX,valY,W,lr,epoch,C,sigma):
    precision = []
    recall = []
    F1 = []
    loss = []
    Gaussian_core = trainX
    trainX = Gaussian_distance(trainX,Gaussian_core,sigma)  #81x80
    valX = Gaussian_distance(valX,Gaussian_core,sigma)  #81x20
    for i in range(0,epoch):
        sigY = trainY>0
        predictY = np.dot(W.transpose(),trainX)
        partialW = np.sum(C * (sigY * (predictY<1)*(-trainX) +(1-sigY)*(predictY>-1)*trainX),1).reshape([81,1]) + 2*W
        W = W - lr * partialW
        if (i % 5 == 0):
            p, r, f, l = val(valX, valY, W, i)
            precision.append(p)
            recall.append(r)
            F1.append(f)
            loss.append(l)

        # if(predictY>=1):
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
W = np.random.rand(80+1,1)

lr = 0.001
epoch = 1000
C = 5
sigma = 1
train(trainX,trainY,valX,valY,W,lr,epoch,C,sigma)
