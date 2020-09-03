import random
import numpy as np
from numpy.ma import exp,sin,cos
from random import *
def tanhgra(x):
    ret = np.zeros(x.shape)
    for i in range(len(x)):
        ret[i] = 4.0/(exp(2*x[i])+exp(-2*x[i])+2)
    return ret    
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
def gradient(features,lables,w):
    theta = -lables*np.dot(features,w)
    logi = 1/(1+exp(-theta))
    return 1.0/len(features)*sum(logi*-lables* features).reshape(len(w),1)

def sgradient(feature,lable,w):
    theta = (-lable*np.dot(feature,w))
    logi = 1/(1+exp(-theta))
    return (logi*-lable*feature).reshape(len(w),1)
def init_w(d,r):
    w = []
    for i in range(len(d)-1):
        w.append(np.random.uniform(-r,r,(d[i]+1,d[i+1])))
        #w.append((i+1)*0.01*np.ones((d[i]+1,d[i+1])))
    return w
def forward(X,w,d):
    s=[];
    x=[];
    for i in range(len(d)-1):
        if (i==0):
            s.append(X.dot(w[i]))
        else :
            s.append(np.dot(np.hstack((1,x[-1])),w[i]))
        x.append(np.tanh(s[-1]))
    return s,x
def backward(X,Y,s,x,w,d):
    gra = [];
    ret = [];
    for i in reversed(range(len(d)-1)):
        if (i==len(d)-2):
            gra = np.array([-2*(Y-x[i])*tanhgra(s[i])]);
        elif (i!=0):
            gra = w[i+1][1:].dot(gra.transpose()).transpose()*tanhgra(s[i])
        elif (i==0):
            gra = w[i+1][1:].dot(gra.transpose()).transpose()*tanhgra(s[i])
        ret.insert(0,gra)    
    return ret
def update(X,w,s,x,gra,ita=0.1):
    for i in range(len(w)):
        if (i!=0):
            #print x[i-1].shape
            #print gra[i].shape
            w[i] = w[i] - ita*(np.array([np.hstack((1,x[i-1]))]).transpose()*gra[i])
        else :
            w[i] = w[i] - ita*np.array([X]).transpose()*gra[i]
    return w

def train(X,Y,sgd,w,d,T,ita):
    #print "w: ",w
    for i in range(T):
        index = sgd[i]
        #print "X: Y",X[index],Y[index]
        s,x = forward(X[index],w,d)
        #print "s: ",s
        #print "x: ",x
        gra = backward(X[index],Y[index],s,x,w,d)
        #print "gra: ",gra
        w = update(X[index],w,s,x,gra,ita)
        #print "w: ",w
def predict(X,Y,W):
    tmp = X;
    for w in W:
        tmp = np.tanh(np.dot(tmp,w))
        tmp = np.hstack((1,tmp))
    #print tmp[1]    
    return tmp[1]*Y<-1e-10
def error_rate(X,Y,w,d):
    err=0
    for i in range(len(X)):
        if predict(X[i],Y[i],w):
            err+=1
    return err*1.0/len(X)
X,Y= loadData("hw4_nnet_train.dat")
testX,testY= loadData("hw4_nnet_test.dat")
err=np.zeros((22,10));
for T in range(10):
    sgd=[]
    for i in range(50000):
        sgd.append(randint(0,len(X)-1))
    #Q14
    for M in [8]:
        ita = 0.01
    #Q13
    #for ita in [0.001,0.01,0.1,1,10]:
        r=0.1
    #Q12
    #for r in [0,0.001,0.01,10,1000]:
    #    M=3    
    #Q11    
    #for M in [1,6,11,16,21]:
        d=[len(X[0])-1,M,3,1]
        w = init_w(d,r)
        train(X,Y,sgd,w,d,50000,ita)
        err[M][T]=error_rate(testX,testY,w,d);
        print ita,r,M,err[M][T]+0.00000001
for M in [1,6,11,16,21]:
    print M
    print np.mean(err[M])
