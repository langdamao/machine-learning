#encoding=utf8
import sys
import numpy as np
import math
from random import *


##
# tree node for storing C&RT model's decision node
# i: feature index
# v: decision-stump threshold value
# s: decision-stump sign ( direction )
# left: left branch TreeNode
# right: right branch TreeNode
class TreeNode:
    def __init__(self, i, v):
        self.index = i
        self.val = v
        self.sign = 0
        self.left = None
        self.right = None

##
# read data from local file
# return with numpy array
def read_input_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        if line.strip()=='': continue
        items = line.strip().split(' ')
        tmp_x = []
        for i in range(0,len(items)-1): tmp_x.append(float(items[i]))
        x.append(tmp_x)
        y.append(float(items[-1]))
    return np.array(x),np.array(y)

##
# input All data ( binary categories in this context )
# learning decision-stump from the data
# splited subdata via learned decision-stump
# return two splited data, index, val, sign
def splited_by_decisionStump(x, y):
    # storeing sorted index via all x's certain feature
    sorted_index = []
    for i in range(0, x.shape[1]):
        sorted_index.append(np.argsort(x[:,i]))
    # learn the best feature for this node's decision stump
    n1 = x.shape[0]/2
    n2 = x.shape[0]-n1
    Branch = float("inf")
    index = -1
    val = 0
    for i in range(0, x.shape[1]):
        # learn decision stump via x[i]
        xi = x[sorted_index[i], i]
        yi = y[sorted_index[i]]
        # minimize cost function of feature i
        b, v = learn_decisionStump(xi, yi)
        # update least impuirty parameter (val,sign)
        if Branch>b:
            Branch = b
            index = i
            val = v
    # spliting data with best feature and it's val & sign
    leftX = x[np.where(x[:,index]<val)]
    leftY = y[np.where(x[:,index]<val)]
    rightX = x[np.where(x[:,index]>=val)]
    rightY = y[np.where(x[:,index]>=val)]
    return leftX, leftY, rightX, rightY, index, val

# learn decision-stump threshold from one feature dimension
def learn_decisionStump(x,y):
    # calculate median of interval
    thetas = np.array([ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ] )
    B = float("inf")
    target_theta = 0.0
    # traversal each median value
    for theta in thetas:
        ly = y[np.where(x<theta)]
        ry = y[np.where(x>=theta)]
        b = ly.shape[0]*calculate_GiniIndex(ly) + ry.shape[0]*calculate_GiniIndex(ry)
        if B>b:
            B = b
            target_theta = theta
    return B, target_theta


## 
# input data ( binary catergories in this context )
# return with Gini Index
def calculate_GiniIndex(y):
    if y.shape[0]==0: return 0
    n1 = sum(y==1)
    n2 = sum(y==-1)
    if (n1+n2)==0: return 0
    return 1.0 - math.pow(1.0*n1/(n1+n2),2) - math.pow(1.0*n2/(n1+n2),2)


## 
# C&RT tree's dfs learning algorithm
# return with learned model within a binary tree
def CART(x, y):
    if x.shape[0]==0: return None # none case
    if calculate_GiniIndex(y)==0: # terminal case ( only one category )
        node = TreeNode(-1, -1)
        node.sign = 1 if y[0]==1 else -1
        return node
    leftX, leftY, rightX, rightY, index, val = splited_by_decisionStump(x,y)
    node = TreeNode(index,val)
    node.left = CART(leftX, leftY)
    node.right = CART(rightX, rightY)
    return node

## Q13
# count internal nodes
def count_internal_nodes(root):
    if root==None: return 0
    if root.left==None and root.right==None: return 0
    print root.index, root.val
    l = 0
    r = 0
    if root.left!=None: 
        l = count_internal_nodes(root.left)
    if root.right!=None: 
        r = count_internal_nodes(root.right)
    return 1 + l + r

## Q15
# predict
def predict(root, x):
    if root.index==-1: return root.sign
    if x[root.index]<root.val:
        return predict(root.left, x)
    else:
        return predict(root.right, x)
# calculate Eout
def calculate_E(model, path):
    x,y = read_input_data(path)
    error_count = 0
    for i in range(0, x.shape[0]):
        error_count = error_count + (1 if predict(model, x[i])!=y[i] else 0)
    return 1.0*error_count/x.shape[0]

## Q16
# Random Forest via Bagging and average Ein(gt)
def randomForest(x, y, T):
    error_rate = 0
    trees = []
    for i in range(0,T):
        xi,yi = naive_sampling(x, y)
        model = CART(xi,yi)
        error_rate += calculate_E(model,"hw3_dectree_train.dat")
        trees.append(model)
    return error_rate/T, trees
# holy shit naive sampling
def naive_sampling(x, y):
    sampleX = np.zeros(x.shape)
    sampleY = np.zeros(y.shape)
    for i in range(0, x.shape[0]):
        index = randint(0, x.shape[0]-1)
        sampleX[i] = x[index]
        sampleY[i] = y[index]
    return sampleX, sampleY

## Q17 Q18
# Ein(G)
def calculate_RF_E(trees, path):
    x,y = read_input_data(path)
    error_count = 0
    for i in range(0, x.shape[0]):
        yp = rf_predict(trees, x[i])
        error_count += 1 if yp!=y[i] else 0
    return 1.0*error_count/x.shape[0]
# random forest predict process
def rf_predict(trees, x):
    positives = 0
    negatives = 0
    for tree in trees:
        yp = predict(tree, x)
        if yp==1:
            positives += 1
        else:
            negatives += 1
    return 1 if positives>negatives else -1

## Q19
# prune to only one branch
def one_branch_CART(x, y):
    if x.shape[0]==0: return None # none case
    if calculate_GiniIndex(y)==0: # terminal case ( only one category )
        node = TreeNode(-1, -1)
        node.sign = 1 if y[0]==1 else -1
        return node
    leftX, leftY, rightX, rightY, index, val = splited_by_decisionStump(x,y)
    node = TreeNode(index, val)
    node.left = TreeNode(-1, -1)
    node.right = TreeNode(-1, -1)
    ly = y[np.where(x[:,index]<val)]
    node.left.sign = 1 if sum(ly==1)>sum(ly==-1) else -1
    node.right.sign = -node.left.sign
    return node

def one_branch_randomForest(x, y, T):
    trees = []
    for i in range(0,T):
        xi,yi = naive_sampling(x, y)
        model = one_branch_CART(xi, yi)
        trees.append(model)
    return trees



def main():
    # x,y = read_input_data("unitTestSplitedByDecisionStump.dat")
    x,y = read_input_data("hw3_dectree_train.dat")
    root = CART(x,y)
    print count_internal_nodes(root)
    print calculate_E(root, "hw3_dectree_train.dat")
    print calculate_E(root, "hw3_dectree_test.dat")
    error_rate,trees = randomForest(x, y, 301)
    print error_rate
    print calculate_RF_E(trees, "hw3_dectree_train.dat")
    print calculate_RF_E(trees, "hw3_dectree_test.dat")
    trees = one_branch_randomForest(x, y, 301)
    print calculate_RF_E(trees, "hw3_dectree_train.dat")
    print calculate_RF_E(trees, "hw3_dectree_test.dat")

if __name__ == '__main__':
    main()
