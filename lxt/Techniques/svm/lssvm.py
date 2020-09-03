import random
import numpy as np
from numpy.ma import exp,sin,cos
def loadData(file):
    f = open(file)
    try:
        lines = f.readlines()
    finally:
        f.close() 
    num = len(lines)
    wide = len(lines[0].strip().split(" "))
    features = np.zeros((num,wide))
    lables = np.zeros((num,1))
    for i,line in enumerate(lines):
        item = line.strip().split(" ")
        features[i][0]=1;
        features[i][1:] = [float(x) for x in item[0:-1]]
        lables[i] = float(item[-1])
    return features,lables
def error_rate(features,lables,wlin):
    W = features.dot(wlin)
    ret=0
    for i in range(len(lables)):
        if (W[i][0]*lables[i][0]<0):
            ret = ret+1;
    return ret*1.0/len(lables)        
def rbf(gamma,x,y):
   return exp(-1*gamma*(np.linalg.norm(x-y,ord=2, axis=None, keepdims=False)))
def getK(X,gamma):
    K = np.zeros((len(X),len(X)))
    for x in range(len(X)):
        for y in range(len(X)):
            K[x][y]=rbf(gamma,X[x],X[y])
    return K        

def getBeta(alpha,K,Y):
    return np.linalg.inv(alpha*np.identity(len(Y))+K).dot(Y)
def error_rate(X,Y,KX,beta,gamma):
    err=0.0
    K = np.zeros((1,len(KX)))
    for i in range(len(X)):
        for j in range(len(KX)):
            K[0][j] = rbf(gamma,X[i],KX[j])
        g = K.dot(beta)
        if (g*Y[i]<0):
            err = err + 1.0
    return err/len(Y)

features,lables = loadData("hw2_lssvm_all.dat")
trainX = features[0:400]
trainY = lables[0:400]
testX = features[400:]
testY = lables[400:]
for gamma in [0.125,2,32]:
    K = getK(trainX,gamma)
    for alpha in [0.001,1,1000]:
        beta = getBeta(alpha,K,trainY)
        error_in = error_rate(trainX,trainY,trainX,beta,gamma)
        error_out = error_rate(testX,testY,trainX,beta,gamma)
        print gamma,alpha,error_in,error_out
