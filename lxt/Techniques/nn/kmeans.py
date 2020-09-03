import random
import numpy as np
from numpy.ma import exp,sin,cos
from random import *
from heapq import *
eps = 1e-10
def loadData(file):
    f = open(file)
    try:
        lines = f.readlines()
    finally:
        f.close() 
    num = len(lines)
    wide = len(lines[0].strip().split(" "))
    features = np.zeros((num,wide))
    #lables = np.zeros((num,1))
    for i,line in enumerate(lines):
        item = line.strip().split(" ")
        features[i]= [float(x) for x in item]
    #    lables[i] = float(item[-1])
    return features
X = loadData("hw4_nolabel_train.dat")
#testX,testY = loadData("hw4_nbor_test.dat")
def sign(x):
    if (x>=0):
        return 1
    else:
        return -1
def predict(x,X,Y,k):
    tmp = np.sum((X-x)*(X-x),axis=1).tolist()
    indexs = map(tmp.index, nsmallest(k, tmp))
    return sign(np.sum([Y[i] for i in indexs]))
def error_rate(u,X,Y):
    err=0.0
    for i in range(len(X)):
        err+=np.sum((X[i]-u[Y[i]])*(X[i]-u[Y[i]]))
    return err/len(Y)
def init_u(k,n,X):
    indexs =  sample(range(0,n-1),k)
    return np.array([X[i] for i in indexs])

def update_y(u,X,y):
    for i in range(len(X)):
        y[i] = np.argmin(np.sum((u-X[i])*(u-X[i]),axis=1))
def update_u(u,X,y):
    ret=[]
    for k in range(len(u)):
        tmp = np.where(y==k)
        ret.append(np.sum(X[tmp],axis=0)*1.0/np.array(tmp).shape[1])
    return np.array(ret)
def change(newu,u):
    return np.sum((newu-u)*(newu-u))
err=[]
k=10
for T in range(500):
    u=init_u(k,len(X),X)
    y=np.array([0]*len(X))
    while(True):
        update_y(u,X,y)
        newu = update_u(u,X,y)
        if (change(newu,u)<eps):
            break
        u = newu
    err.append(error_rate(u,X,y))
    print T,err[-1]
print "end"
print np.mean(err)
