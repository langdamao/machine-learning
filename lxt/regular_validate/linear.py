import random
import numpy as np

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

def calwreg(X,Y,lam):
    return np.linalg.inv(X.T.dot(X)+lam*np.mat(np.identity(len(X[0])))).dot(X.T).dot(Y)
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
#Q13
#lam = 10.0
#features,lables = loadData("hw4_train.dat");
#wreg = calwreg(features,lables,lam)
#e_in = error_rate(features,lables,wreg)
#test_features,test_lables = loadData("hw4_test.dat")
#e_out = error_rate(test_features,test_lables,wreg)
#print wreg
#print e_in,e_out
#Q14
#features,lables = loadData("hw4_train.dat");
#test_features,test_lables = loadData("hw4_test.dat")
#for lamlog in range(-10,3):
#    lam = pow(10,lamlog)
#    wreg = calwreg(features,lables,lam)
#    e_in = error_rate(features,lables,wreg)
#    e_out = error_rate(test_features,test_lables,wreg)
#    print lam, e_in,e_out
#Q16
#features,lables = loadData("hw4_train.dat");
#train_features = features[:120]
#train_lables = lables[:120]
#val_features = features[-80:]
#val_lables= lables[-80:]
#test_features,test_lables = loadData("hw4_test.dat")
#for lamlog in range(-10,3):
#    lam = pow(10,lamlog)
#    wreg = calwreg(train_features,train_lables,lam)
#    e_in = error_rate(train_features,train_lables,wreg)
#    e_val = error_rate(val_features,val_lables,wreg)
#    e_out = error_rate(test_features,test_lables,wreg)
#    wreg = calwreg(features,lables,lam)
#    e_in_tot = error_rate(features,lables,wreg)
#    e_out_tot = error_rate(test_features,test_lables,wreg)
#    print lamlog,lam, e_in,e_val,e_out,e_in_tot,e_out_tot
# Q19
features,lables = loadData("hw4_train.dat");
train_features = features[:120]
train_lables = lables[:120]
val_features = features[-80:]
val_lables= lables[-80:]
test_features,test_lables = loadData("hw4_test.dat")
for lamlog in range(-10,3):
    lam = pow(10,lamlog)
    e_val=[]
    for i in range(0,5):
        val_features = features[i*40:i*40+40]
        val_lables = lables[i*40:i*40+40]
        train_features = np.append(features[0:i*40],features[i*40+40:len(features)]).reshape(160,len(features[0]))
        train_lables= np.append(lables[0:i*40],lables[i*40+40:len(lables)]).reshape(160,len(lables[0]))
        wreg = calwreg(train_features,train_lables,lam)
        e_in = error_rate(train_features,train_lables,wreg)
        e_val.append(error_rate(val_features,val_lables,wreg))
        e_out = error_rate(test_features,test_lables,wreg)
    wreg = calwreg(features,lables,lam)
    e_in_tot = error_rate(features,lables,wreg)
    e_out_tot = error_rate(test_features,test_lables,wreg)
    print lamlog,lam, e_in_tot,e_out_tot
    print lamlog,lam, sum(e_val), len(e_val),sum(e_val)/len(e_val)
