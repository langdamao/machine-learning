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
features,lables = loadData("train")
def gradient(features,lables,w):
    theta = -lables*np.dot(features,w)
    logi = 1/(1+exp(-theta))
    return 1.0/len(features)*sum(logi*-lables* features).reshape(len(w),1)

def sgradient(feature,lable,w):
    theta = (-lable*np.dot(feature,w))
    logi = 1/(1+exp(-theta))
    return (logi*-lable*feature).reshape(len(w),1)
w = np.zeros((21,1))
#Q18
#eta = 0.001
#Q19
#eta = 0.01
#for i in range(2000):
#    if (i%100==0):
#        print i
#    g = gradient(features,lables,w)
#    w = w - eta*g
#Q20
eta=0.001
for i in range(2000):
    if (i%100) ==0:
        print i
    index = i%len(lables);
    g = sgradient(features[index],lables[index],w)
    w = w - eta*g

features,lables = loadData("test")
err = error_rate(features,lables,w)
print(w)
print("Q18 :",err)
