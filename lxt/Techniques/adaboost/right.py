import numpy as np 

def getData(file):
    f = open(file)
    data = f.readlines()
    xs = []
    ys = []
    for line in data:
        d = line.split()
        x = np.array([float(d[0]),float(d[1])])
        y = float(d[2])
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys) 

def weightedErrorRate(x,y,s,theta,h,dimension,weights):
    error = 0
    for i in range(len(x)):
        if y[i] != h(x[i][dimension],s,theta):
            error += weights[i] 
    return error/np.sum(weights)

def hFunc(x,s,theta):
    if s:
        return sign(x-theta)
    else:
        return -sign(x-theta)

def sign(v):
    if v < 0:
        return -1
    else:
        return 1

def updateWeights(x,y,s,theta,dim,epsilon,h,weights):
    for i in range(len(x)):
        if y[i] != h(x[i][dim],s,theta):
            weights[i] *= epsilon
        else:
            weights[i] /= epsilon

def trainDecisionStump(x,y,weights):
    dimensions = len(x[0])
    E_in = 1
    best_s = True
    best_theta = 0
    best_dim = 0
    for dim in range(dimensions):
        thetas = np.sort(x[:,dim]) 
        ss = [True,False] 
        for theta in thetas:
            for s in ss:
                E = weightedErrorRate(x,y,s,theta,hFunc,dim,weights)
                if E < E_in:
                    E_in = E
                    best_s = s
                    best_theta = theta
                    best_dim = dim
    return best_s,best_theta,best_dim,E_in

def adaboostStump(trainX, trainY, trainFunc, weights, T):
    alphas = []
    g_funcs = []
    min_error = 10000
    for i in range(T):
        s,theta,dim,error = trainFunc(trainX,trainY,weights)
        g_funcs.append([s,theta,dim])
        epsilon = np.sqrt((1-error)/error)
        alphas.append(np.log(epsilon))

        # if i %50 == 49:
        #     print('iteration:',i+1)
        min_error = min(error,min_error)

        print('iteration ',i+1,':','sum of weights:',np.sum(weights))
        print('error:',error)
        updateWeights(trainX,trainY,s,theta,dim,epsilon,hFunc,weights)

    print('minimum of error is:',min_error)

    
    return alphas,g_funcs

def applyRes(alphas,g_funcs,h,x):
    res = 0
    for i in range(len(alphas)):
        s,theta,dim = g_funcs[i]
        res += alphas[i]*h(x[dim],s,theta)
    return sign(res)

def boostedError(alphas,g_funcs,h,x,y):
    err = 0.0
    for i in range(len(x)):
        if applyRes(alphas,g_funcs,h,x[i]) != y[i]:
            err += 1
    return err/len(y)

def gError(x,y,h,s,theta,dim):
    err = 0.0
    for i in range(len(y)):
        if h(x[i][dim],s,theta) != y[i]:
            err += 1
    return err/len(y)

def main():
    trainX, trainY = getData('hw2_adaboost_train.dat')
    testX, testY = getData('hw2_adaboost_test.dat')
    weights = [1.0/len(trainY)]*len(trainY)
    weights = np.array(weights)
    T = 300
    alphas,g_funcs = adaboostStump(trainX,trainY,trainDecisionStump,weights,T)

    g1_e_in = gError(trainX,trainY,hFunc,g_funcs[0][0],g_funcs[0][1],g_funcs[0][2])
    print('E_in of g1:',g1_e_in)

    g1_e_out = gError(testX,testY,hFunc,g_funcs[0][0],g_funcs[0][1],g_funcs[0][2])
    print('E_out of g1:',g1_e_out)

    gt_e_in = boostedError(alphas,g_funcs,hFunc,trainX,trainY)
    print('E_in of gT:',gt_e_in)

    gt_e_out = boostedError(alphas,g_funcs,hFunc,testX,testY)
    print('E_out of gT:',gt_e_out)

if __name__ == '__main__':
     main()
