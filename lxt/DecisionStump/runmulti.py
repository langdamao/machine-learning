# -- coding:utf-8 --
import random 
import math
import numpy as np
def sign(x):
    if (x>0): 
        return 1
    else:
        return -1
def generateData(dataMulti,y,file):
    dataSet = open(file, 'r').readlines()
    n = len(dataSet)
    dataMulti = np.zeros((n, 9))
    y = np.zeros(n)
    for i, item in enumerate(dataSet):
        each = item.strip().split()
        dataMulti[i] = [float(a) for a in each[:-1]]
        y[i] = int(each[-1])
    return (dataMulti,y)
def getErr(data,y,theta):
    ret=0;
    for i in range(0,len(data)):
        tmp = sign(data[i]-theta)
        if (tmp != y[i]):
	    ret = ret+1
    return ret*1.0/len(data)
def findThetas(data,y,sumin,sumout):
    thetaList =[-1.0]
    datax = sorted(data)
    datax.append(1.0)
    for i in range(0,len(data)):
	thetaList.append((datax[i]+datax[i+1])/2.0)
    err_in = 1e9*1.0
    err_out = 1e9*1.0
    for i in range(0,len(data)+1):
	err = getErr(data,y,thetaList[i])
	if (err_in>err):
	    err_in = err;
	    err_out = 0.5+0.3*(math.fabs(thetaList[i])-1);
            theta = thetaList[i]
            s=1
	if (err_in > 1-err):
            err_in = 1-err;
            err_out = 0.5-0.3*(1-math.fabs(thetaList[i])-1);
            theta = thetaList[i]
            s=-1
    return theta,s,err_in 

sumin=0.0
sumout=0.0
n = 20
dataMulti=[[]]
y = []
ans=[]
dataMulti,y = generateData(dataMulti,y,"train")
for i in range(0,len(dataMulti[0])):
    data = [x[i] for x in dataMulti]
    ans.append(findThetas(data,y,sumin,sumout))
print ans    
err_ins=[x[2] for x in ans]
err_in=np.min(err_ins)
dim=np.argmin(err_ins)
theta=ans[dim][0]
s=ans[dim][1]
dataMulti=[[]]
y = []
ret =0
dataMulti,y = generateData(dataMulti,y,"test")
data = [x[dim] for x in dataMulti]
for i in range(0,len(data)):
    tmp = sign(sign(data[i]-theta)*s);
    if (tmp !=y[i]) :
       ret = ret+1
print ret       
print err_in, ret*1.0/len(y)
