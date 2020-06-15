# -- coding:utf-8 --
import random 
import math
def sign(x):
    if (x>0): 
        return 1
    else:
        return -1
def generateData(data,y,n):
    noise=[]
    for i in range(0,n):
        data.append(random.uniform(-1,1))
        noise.append(random.uniform(-0.2,0.8))
        y.append(sign(noise[i])*sign(data[i]))
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
	if (err_in > 1-err):
            err_in = 1-err;
            err_out = 0.5-0.3*(1-math.fabs(thetaList[i])-1);
    sumin = sumin+err_in
    sumout = sumout+err_out
    return sumin,sumout

sumin=0.0
sumout=0.0
n = 20
for i in range(1,5000):
    data=[]
    y = []
    generateData(data,y,n)
    sumin,sumout = findThetas(data,y,sumin,sumout)
print sumin/5000, sumout/5000
