import random
import numpy as np
import math
from numpy.ma import exp,sin,cos
def loadData(file):
    f = open(file)
    try:
        lines = f.readlines()
    finally:
        f.close() 
    num = len(lines)
    wide = len(lines[0].strip().split(" "))
    features = np.zeros((num,wide-1))
    lables = np.zeros((num,1))
    for i,line in enumerate(lines):
        item = line.strip().split(" ")
        features[i] = [float(x) for x in item[0:-1]]
        lables[i] = float(item[-1])
    return features,lables
def error_rate(features,lables,wlin):
    W = features.dot(wlin)
    ret=0
    for i in range(len(lables)):
        if (W[i][0]*lables[i][0]<0):
            ret = ret+1;
    return ret*1.0/len(lables)        
def g(x,theta,s):
    if (x-theta)*s>= 0 :
        return 1
    else:
        return -1
       
def get_error_rate(i,theta,s,u,features,lables):
    ret=[]
    err=0.0
    for j in range(len(features)):
        if (g(features[j][i],theta,s)*lables[j] <0 ):
            err = err+u[j]
            ret.append(-1)
        else: 
            ret.append(1)    
    return err/np.sum(u),ret;
def error_rate(G,alpha,features,lables):
    err=0.0
    ret=[]
    for j in range(len(features)):
        if  np.sum([g(features[j][G[k][0]],G[k][1],G[k][2])*alpha[k] for k in range(len(G))])*lables[j]<0:
            err = err+1.0
            ret.append(-1)
        else: 
            ret.append(1)    
    return err/N,ret;
def update(u,err,errlist):
    t = np.sqrt((1.0-err)/err)
    ret=[]
    for i in range(len(u)):
        if errlist[i]==1:
            ret.append(u[i]/t)
        else:
            ret.append(u[i]*t)
    
    return ret
def boosting(u,features,lables):
    errs = []
    for i in range(wide):
        for theta in thetas[i]:
            for s in [1,-1]:
                err,errlist = get_error_rate(i,theta,s,u,features,lables)    
                errs.append([err,i,theta,s]);
    minindex=np.argmin(errs, axis=0)[0]           
    return errs[minindex][1:]

features,lables = loadData("hw2_adaboost_train.dat")
wide = len(features[0])
minValue = np.min(features)-100
N = len(lables)
u = [1.0/N for i in range(N)]
G = []
alpha = []
sortedlist=[]
thetas = [[minValue] * N] * wide
for i in range(wide):
    sortedlist.append([x for x in features[:,i]])
    sortedlist[i].sort()
    print sortedlist[i]
    for j in range(len(sortedlist[i])):
        if (j!=0): 
            thetas[i][j] = (sortedlist[i][j]+sortedlist[i][j-1])/2
print thetas
errs=[]
for t in range(300):
    print t,"---------------------------------"
    i,theta,s = boosting(u,features,lables)
    G.append([i,theta,s])
    err,errlist = get_error_rate(i,theta,s,u,features,lables)
    u = update(u,err,errlist)
    alpha.append(math.log(np.sqrt((1.0-err)/err)))
    print err,np.sum(u)
    errs.append(err)
    #print error_rate(G,[1],features,lables)
print min(errs)    
err = error_rate(G,alpha,features,lables)
print err
features,lables = loadData("hw2_adaboost_test.dat")
wide = len(features[0])
minValue = np.min(features)-100
N = len(lables)
err = error_rate(G,alpha,features,lables)
print err
