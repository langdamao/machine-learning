#encoding=utf8
import sys
import numpy as np
import math
from random import *

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
# initialize weight matrix
# input neural network structure & initilizing uniform value range (both low and high)
# each layer's bias need to be added
# return with inialized W
def init_W(nnet_struct, w_range):
    W = []
    for i in range(1,len(nnet_struct)):
        tmp_w = np.random.uniform(w_range['low'], w_range['high'], (nnet_struct[i-1]+1,nnet_struct[i]) )
        W.append(tmp_w)
    return W

## 
# randomly pick sample from raw data for Stochastic Gradient Descent
# T indicates the iterative numbers
# return with data for each SGD iteration
def pick_SGD_data(x, y, T):
    sgd_x = np.zeros((T,x.shape[1]))
    sgd_y = np.zeros(T)
    for i in range(T):
        index = randint(0, x.shape[0]-1)
        sgd_x[i] = x[index]
        sgd_y[i] = y[index]
    return sgd_x, sgd_y

## 
# forward process
# calculate each neuron's output
def forward_process(x, y, W):
    ret = []
    #print W[0].shape
    #print W[1].shape
    pre_x = np.hstack((1,x))
    for i in range(len(W)):
        pre_x = np.tanh(np.dot(pre_x, W[i]))
        ret.append(pre_x)
        pre_x = np.hstack((1,pre_x))
    return ret

##
# backward process
# calcultae the gradient of error and each neuron's input score
def backward_process(x, y, neuron_output, W):
    ret = []
    L = len(neuron_output)
    # print neuron_output[0].shape, neuron_output[1].shape
    # Output layer
    score = np.dot( np.hstack((1, neuron_output[L-2])), W[L-1])
    # print score
    # print score.shape
    gradient = np.array( [-2 * (y-neuron_output[L-1][0]) * tanh_gradient(score)] )
    # print gradient
    # print gradient.shape
    ret.insert(0, gradient)
    # Hidden layer 
    for i in range(L-2,-1,-1):
        if i==0:
            score = np.dot(np.hstack((1, x)),W[i])
            # print score.shape
            # print gradient.shape
            # print W[1][1:].transpose().shape
            # print score
            gradient = np.dot(gradient, W[1][1:].transpose()) * tanh_gradient(score)
            # print gradient
            # print gradient.shapeq
            ret.insert(0, gradient)
        else:
            score = np.dot(np.hstack((1,neuron_output[i-1])),W[i])
            # print score.shape
            # print gradient.shape
            # print W[i+1][1:].transpose().shape
            # print "......"
            gradient = np.dot(gradient , W[i+1][1:].transpose()) * tanh_gradient(score)
            # print gradient.shape
            # print "======"
            ret.insert(0, gradient)
    return ret

# give a numpy array
# boardcast tanh gradient to each element
def tanh_gradient(s):
    ret = np.zeros(s.shape)
    for i in range(s.shape[0]):
        ret[i] = 4.000001 / (math.exp(2*s[i])+math.exp(-2*s[i])+2)
    return ret


##
# update W with Gradient Descent
def update_W_withGD(x, neuron_output, gradient, W, ita):
    ret = []
    L = len(W)
    # print "L:"+str(L)
    # print neuron_output[0].shape, neuron_output[1].shape
    # print gradient[0].shape, gradient[1].shape
    # print W[0].shape, W[1].shape
    # print np.hstack((1,x)).transpose().shape
    # print gradient[0].shape
    ret.append( W[0] - ita * np.array([np.hstack((1,x))]).transpose() * gradient[0] )
    for i in range(1, L, 1):
        ret.append( W[i] - ita * np.array([np.hstack((1,neuron_output[i-1]))]).transpose() * gradient[i] )
    # print len(ret)
    return ret

## 
# calculate Eout
def calculate_E(W, path):
    x,y = read_input_data(path)
    error_count = 0
    for i in range(x.shape[0]):
        if predict(x[i],y[i],W):
            error_count += 1
    return 1.000001*error_count/x.shape[0]

def predict(x, y, W):
    y_predict = x
    for i in range(0, len(W), 1):
        y_predict = np.tanh( np.dot( np.hstack((1,y_predict)), W[i] ) )
    y_predict = 1 if y_predict>0 else -1
    return y_predict!=y

##
# Q11
def Q11(x,y):
    R = 20 # repeat time
    Ms = { 6, 16 } # hidden units
    M_lowests = {}
    for M in Ms: M_lowests[M] = 0
    for r in range(R):
        T = 50000
        ita = 0.1
        min_M = -1
        E_min = float("inf")
        for M in Ms:
            sgd_x, sgd_y = pick_SGD_data(x, y, T)
            nnet_struct = [ x.shape[1], M, 1 ]
            # print nnet_struct
            w_range = {}
            w_range['low'] = -0.1
            w_range['high'] = 0.1
            W = init_W(nnet_struct, w_range)
            # for i in range(len(W)):
            #    print W[i]
            # print sgd_x,sgd_y
            for t in range(T):
                neuron_output = forward_process(sgd_x[t], sgd_y[t], W)
                # print sgd_x[t],sgd_y[t]
                # print W
                # print neuron_output
                error_neuronInputScore_gradient = backward_process(sgd_x[t], sgd_y[t], neuron_output, W)
                # print error_neuronInputScore_gradient
                W = update_W_withGD(sgd_x[t], neuron_output, error_neuronInputScore_gradient, W, ita)
            E = calculate_E(W,"test.dat")
            print str(r)+":::"+str(M)+":"+str(E)
            M_lowests[M] += E
    for k,v in M_lowests.items():
        print str(k)+":"+str(v)

##
# Q12
def Q12(x,y):
    ita = 0.1
    M = 3
    nnet_struct = [ x.shape[1], M, 1 ]
    Rs = { 0.001, 0.1 }
    R_lowests = {}
    for R in Rs: R_lowests[R] = 0
    N = 40
    T = 30000
    for i in range(N):
        for R in Rs:
            sgd_x, sgd_y = pick_SGD_data(x, y, T)
            w_range = {}
            w_range['low'] = -1*R
            w_range['high'] = R
            W = init_W(nnet_struct, w_range)
            for t in range(T):
                neuron_output = forward_process(sgd_x[t], sgd_y[t], W)
                error_neuronInputScore_gradient = backward_process(sgd_x[t], sgd_y[t], neuron_output, W)
                W = update_W_withGD(sgd_x[t], neuron_output, error_neuronInputScore_gradient, W, ita)
            E = calculate_E(W, "test.dat")
            print str(R)+":"+str(E)
            R_lowests[R] += E
    for k,v in R_lowests.items():
        print str(k)+":"+str(v)

## 
# Q13
def Q13(x,y):
    M = 3
    nnet_struct = [ x.shape[1], M, 1 ]
    itas = {0.001,0.01,0.1}
    ita_lowests = {}
    for ita in itas: ita_lowests[ita] = 0
    N = 20
    T = 20000
    for i in range(N):
        for ita in itas:
            sgd_x, sgd_y = pick_SGD_data(x, y, T)
            w_range = {}
            w_range['low'] = -0.1
            w_range['high'] = 0.1
            W = init_W(nnet_struct, w_range)
            for t in range(T):
                neuron_output = forward_process(sgd_x[t], sgd_y[t], W)
                error_neuronInputScore_gradient = backward_process(sgd_x[t], sgd_y[t], neuron_output, W)
                W = update_W_withGD(sgd_x[t], neuron_output, error_neuronInputScore_gradient, W, ita)
            E = calculate_E(W, "test.dat")
            print str(ita)+":"+str(E)
            ita_lowests[ita] += E
    for k,v in ita_lowests.items():
        print str(k)+":"+str(v)

##
# Q14
def Q14(x,y):
    T = 50000
    ita = 0.01
    E_total = 0
    R = 10
    for i in range(R):
        nnet_struct = [ x.shape[1], 8, 3, 1 ]
        w_range = {}
        w_range['low'] = -0.1
        w_range['high'] = 0.1
        W = init_W(nnet_struct, w_range)
        sgd_x, sgd_y = pick_SGD_data(x, y, T)
        for t in range(T):
            neuron_output = forward_process(sgd_x[t], sgd_y[t], W)
            error_neuronInputScore_gradient = backward_process(sgd_x[t], sgd_y[t], neuron_output, W)
            W = update_W_withGD(sgd_x[t], neuron_output, error_neuronInputScore_gradient, W, ita)    
        E = calculate_E(W, "test.dat")
        print E
        E_total += E
    print E_total*1.0/R


def main():
    x,y = read_input_data("train.dat")
    # print x.shape, y.shape
    #Q11(x, y)
    #Q12(x, y)
    Q13(x, y)
    #Q14(x, y)





if __name__ == '__main__':
    main()
