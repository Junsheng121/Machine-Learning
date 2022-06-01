import numpy as np
def cov(x,y):
    return np.sum((x - np.mean(x))*(y-np.mean(y)))
def cov_matrix(X):
    n = X.shape[0]
    matrix = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cov(X[:,i],X[:,j])

    return matrix



def initdataset(dimen,num):
    return np.random.rand(dimen,num)

data = initdataset(20,80)
conv = cov_matrix(data)   #20x20
eig2,U = np.linalg.eig(np.dot(conv,conv.transpose()))
_,V = np.linalg.eig(np.dot(conv.transpose(),conv))
# print(np.sqrt(eig2))
total = np.sum(eig2)
cur = 0
k = 0
percent = 0.99
for i in range(0,eig2.shape[0]):
    cur += eig2[i]
    if(cur >= total*percent):
        k=i
        break
U_comp = U[:,0:k]
data_comp = np.dot(U_comp.transpose(),data)
print (data_comp.shape)

