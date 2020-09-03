import random
import numpy as np
from numpy.ma import exp,sin,cos
eps=1e-10
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
def g(x,i,theta,s):
    if (s*(x[i]-theta)>=0):
        return 1
    else :
        return -1
class Tree:
    def __init__(self,rt):
        self.rt= rt
    def countInNode(self,node):
        if (node.lc==None and node.rc==None):
            return 0
        ret=1
        print node.i,node.theta,node.s
        if (node.lc!=None):
            ret=ret+self.countInNode(node.lc);
        if (node.rc!=None):
            ret=ret+self.countInNode(node.rc);
        return ret
    def p(self,x,node):
        if (node.lc==None and node.rc==None):
            return node.s
        if (g(x,node.i,node.theta,1)>0):
            return self.p(x,node.lc)
        else:
            return self.p(x,node.rc)
    def predict(self,x):
        return self.p(x,self.rt)

        
class Node:
    def __init__(self,i,theta,s):
        self.i=i;
        self.theta=theta
        self.s=s
        self.lc=None
        self.rc=None
def error_rate(tree, X,Y):
    err=0
    for i in range(len(X)):
        if (Y[i]*tree.predict(X[i])<0):
            err=err+1
    return err*1.0/len(X)

def to01(x):
    if (x>0) :
        return 1
    else :
        return 0
def getTheta(X):
    ret = np.zeros((len(X[0]),len(X)-1)) 
    for i in range(len(X[0])):
        tmp = [x  for x in X[:,i]]
        tmp.sort()
        for j in range(len(tmp)-1):
            ret[i][j] = (tmp[j]+tmp[j+1])/2.0
    return ret
def getGini(X,Y,i,theta):
    N=len(X)
    n = np.zeros((2,2))
    ret=0.0
    for k in range(len(X)):
        ii = to01( g(X[k],i,theta,1))
        j = to01(Y[k])
        n[ii][j] = n[ii][j]+1.0
    for k in range(2):
        nn = n[k][0]+n[k][1]+0.0
        if (nn==0):
            ret = ret+N
        else:
            ret = ret+(1.0-(n[k][0]/nn)*(n[k][0]/nn)-(n[k][1]/nn)*(n[k][1]/nn))*nn
    return ret
def separte(X,Y,i,theta):
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    for k in range(len(X)):
        ii = g(X[k],i,theta,1)
        if (ii>0):
            X1.append(X[k])
            Y1.append(Y[k])
        else :
            X2.append(X[k])
            Y2.append(Y[k])
    return np.array(X1),np.array(Y1),np.array(X2),np.array(Y2)
def theSame(X,Y):
    ret=True
    for i in range(len(X)-1):
        if (Y[i]*Y[i+1]<0):
            ret= False
    if (ret):
        return True
    ret=True
    for i in range(len(X)-1):
        for j in range(len(X[i])):
            if (abs(X[i][j]-X[i+1][j])>eps):
                return False;
    return ret

def dectree(X,Y,dep,max_depth=1000):
    #print dep,max_depth
    if (dep>=max_depth or theSame(X,Y)):
        num = np.sum(Y == 1)
        node = Node(0,0,-1)
        if (num>=(len(Y)+1)/2):
            node = Node(0,0,1)
        return node
    thetas = getTheta(X)
    ginis=np.zeros((len(thetas),len(thetas[0])))
    for i in range(len(thetas)):
        for j in range(len(thetas[i])):
            ginis[i][j]=getGini(X,Y,i,thetas[i][j])
    index = np.unravel_index(ginis.argmin(), ginis.shape)
    i = index[0]
    theta = thetas[index[0]][index[1]]
    #print i,theta,ginis[index[0]][index[1]]
    X1,Y1,X2,Y2=separte(X,Y,i,theta)
    #print X1,Y1
    #print "--------------"
    #print X2,Y2
    #print i,theta,ginis[index[0]][index[1]]
    node=Node(i,theta,0)
    #print i,theta,len(X1),len(X2)
    node.lc = dectree(X1,Y1,dep+1,max_depth)
    #print i,theta,len(X1),len(X2)
    node.rc = dectree(X2,Y2,dep+1,max_depth)
    #print i,theta,node.lc.i,node.lc.theta,node.rc.i,node.rc.theta
    return node
class RF:
    trees=[]
    def __init__(self):
        trees=[]
    def addTree(self,tree):
        self.trees.append(tree)
    def predict(self,x):
        ret=0
        for tree in self.trees:
            ret = ret+tree.predict(x)
        if (ret>=0) :
            return 1
        else :
            return -1
def getRandom(X,Y,N):
    n = len(X)
    index = [random.randint(0,n-1) for _ in range(N)]
    XX = [X[i] for i in index]
    YY = [Y[i] for i in index]
    return np.array(XX),np.array(YY)
        
def trainForest(X,Y,max_depth=1000):
    err=[]
    N=len(X)
    forest=RF()
    for t in range(300):
        if (t%10==0): 
            print t
        XX,YY=getRandom(X,Y,N)
        # ping XX,YY
        tree = Tree(dectree(XX,YY,0,max_depth))
        er = error_rate(tree,X,Y)
        print er
        err.append(er)
        forest.addTree(tree)
    print(np.mean(err))
    return forest
def error_rate(forest,X,Y):
    N=len(X)
    err=0
    for i in range(N):
        if Y[i]*forest.predict(X[i])<0:
            err=err+1
    print err*1.0/N
    return err*1.0/N


#Q16
features,lables = loadData("hw3_dectree_train.dat")
testX,testY= loadData("hw3_dectree_test.dat")
ein=[]
eout=[]
for t in range(2):
    print "-----t:",t
    forest = trainForest(features,lables)
    ein.append(error_rate(forest,features,lables))
    eout.append(error_rate(forest,testX,testY))
print ein,eout
print np.mean(ein),np.mean(eout)
#Q18
#ein=[]
#eout=[]
#for t in range(10):
#    print "-----t:",t
#    forest = trainForest(features,lables,1)
#    ein.append(error_rate(forest,features,lables))
#    eout.append(error_rate(forest,testX,testY))
#print ein,eout
#print np.mean(ein),np.mean(eout)

#print test
#print test1


