import random
import numpy as np
from numpy.ma import exp,sin,cos
from random import *
from heapq import *
def loadData(file):
    f = open(file)
    try:
        lines = f.readlines()
    finally:
        f.close() 
    num = len(lines)
    wide = len(lines[0].strip().split(" "))-1
    features = np.zeros((num,wide))
    lables = np.zeros((num,1))
    for i,line in enumerate(lines):
        item = line.strip().split(" ")
        features[i]= [float(x) for x in item[0:-1]]
        lables[i] = float(item[-1])
    return features,lables
X,Y = loadData("hw4_nbor_train.dat")
testX,testY = loadData("hw4_nbor_test.dat")
def sign(x):
    if (x>=0):
        return 1
    else:
        return -1
def predict(x,X,Y,k):
    tmp = np.sum((X-x)*(X-x),axis=1).tolist()
    indexs = map(tmp.index, nsmallest(k, tmp))
    return sign(np.sum([Y[i] for i in indexs]))
def error_rate(testX,testY,X,Y,k):
    err=0
    for i in range(len(testX)):
        if (testY[i]*predict(testX[i],X,Y,k)<0):
            err+=1
    return err*1.0/len(testY)
k=5
print error_rate(X,Y,X,Y,k)
print error_rate(testX,testY,X,Y,k)

