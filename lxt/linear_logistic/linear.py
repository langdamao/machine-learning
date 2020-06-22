import random
import numpy as np
def f(x):
    if (x[1]*x[1]+x[2]*x[2]-0.6>=0):
        return 1
    return -1
def gerateInput(num=1000):
    features = np.zeros((num,3));
    lables = np.zeros((num,1));

    for i in range(num):
        features[i][0]=1;
        features[i][1]= random.uniform(-1,1)
        features[i][2] = random.uniform(-1,1)
        lables[i][0] = f(features[i]);
        k = random.uniform(0,1)
        if (k<0.1): 
            lables[i][0] = -lables[i][0]
    return features,lables

def calwlin(X,Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
def error_rate(features,lables,wlin):
    W = features.dot(wlin)
    ret=0
    for i in range(len(lables)):
        if (W[i][0]*lables[i][0]<0):
            ret = ret+1;
    return ret*1.0/len(lables)        
def transform(features):
    num = len(features)
    tra_features = np.zeros((num,6))
    for i in range(num):
        tra_features[i]=[features[i][0],features[i][1],features[i][2],features[i][1]*features[i][2],features[i][1]*features[i][1],features[i][2]*features[i][2]]
    return tra_features
# Q13
#err=[]
#for i in range(1000):
#    features,lables = gerateInput(1000)
#    wlin = calwlin(features,lables)
#    err.append(error_rate(features,lables,wlin))
#avg_err = sum(err) / (len(err) * 1.0)
#print("Q13 :",avg_err)
#Q14
#err = float("inf")
#w=[[]]
#for i in range(1000):
#     features,lables = gerateInput(1000)
#     tra_features = transform(features)
#     wlin = calwlin(tra_features,lables)
#     tmp = error_rate(tra_features,lables,wlin)
#     if (tmp<err):
#         err = tmp
#         w=wlin
#print("Q14:")
#print w
#Q15
err=[]
wlin=[[-1],[-0.05],[0.08],[0.13],[1.5],[1.5]]
for i in range(2000):
    features,lables = gerateInput(1000)
    tra_features = transform(features)
    err.append(error_rate(tra_features,lables,wlin))
avg_err = sum(err) / (len(err) * 1.0)
print("Q15 :",avg_err)
