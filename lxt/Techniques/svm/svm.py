import numpy as np
from sklearn.svm import SVC

def flable(x,k):
    if (abs(x-k)<1e-10):
        return 1
    else :
        return -1 
def loadData(file,k):
    f = open(file)
    try:
        lines = f.readlines()
    finally:
        f.close() 
    num = len(lines)
    wide = len(lines[0].strip().split())-1
    features = np.zeros((num,wide))
    lables = np.zeros(num)
    for i,line in enumerate(lines):
        item = line.strip().split()
        lables[i]=flable(eval(item[0]),k);
        features[i] = [ eval(x) for x in item[1:]]
    return features,lables
def calerr(clf,X,Y):
    yhat=np.array(clf.predict(X))
    acc = sum(yhat==Y)
    return 1-acc*1.0/len(Y)


#Q15
#X,Y = loadData("features.train",0)
#TX,TY = loadData("features.test",0)
#clf = SVC(C=0.01,kernel='linear')
#print(clf.coef_)
#print(np.sum(clf.coef_))
#Q16-17
#for k in range(0,10,2):
#    print k
#    X,Y = loadData("features.train",k)
#    TX,TY = loadData("features.test",k)
#    clf = SVC(C=0.01,kernel='poly',degree=2,gamma=1,coef0=1)
#    clf.fit(X,Y);
#    print(clf.dual_coef_)
#    print(np.sum([abs(x) for x in clf.dual_coef_]))
#    err = calerr(clf,X,Y)
#    print(err)
#Q18-19
#X,Y = loadData("features.train",0)
#TX,TY = loadData("features.test",0)

#Cs = [0.001,0.01,0.1,1,10]
#gammas =[1,10,100,1000,10000]
#for gamma in gammas:
#    clf = SVC(C=0.1,kernel='rbf',gamma=gamma)
#    clf.fit(X,Y);
#    print gamma
#    #print(np.sum([x*x for x in clf.dual_coef_]))
#    #print(clf.n_support_)
#    #print(np.sum(clf.n_support_))
#    print(calerr(clf,TX,TY))

#Q20
X,Y = loadData("features.train",0)
TX,TY = loadData("features.test",0)

Cs = [0.001,0.01,0.1,1,10]
gammas =[1,10,100,1000,10000]
choose = []
for i in range(0,100):
    print i
    permutation = np.random.permutation(len(X))
    trainIndex = permutation[1000:]
    valIndex = permutation[0:1000]
    TX = X[trainIndex]
    TY = Y[trainIndex]
    VX = X[valIndex]
    VY = Y[valIndex]
    err=[]
    for gamma in gammas:
        clf = SVC(C=0.1,kernel='rbf',gamma=gamma)
        clf.fit(TX,TY);
        print gamma
        #print(np.sum([x*x for x in clf.dual_coef_]))
        #print(clf.n_support_)
        #print(np.sum(clf.n_support_))
        err.append(calerr(clf,VX,VY))
    choose.append(np.where(err==min(err))[0][0])   
print(choose)    
print(gammas[max(choose,key=choose.count)])

#print(clf.support_)
#print(clf.support_vectors_)
#print(clf.coef_)
#print(clf.intercept_)
#print(clf.n_support_)







