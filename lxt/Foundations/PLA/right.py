import sys
import numpy as np
import random as rd
#copy from https://blog.csdn.net/rikichou/article/details/78226036
### learning rate
rate = 1

def pla_error_rate(features, lables, parameter_vector):
    length = len(features)

    right = 0
    wrong = 0

    for i in range(length):
        if lables[i][0]*(np.dot(features[i], parameter_vector)[0]) <= 0:
            wrong += 1
        else:
            right += 1
    return float(wrong)/float(length)

def pla_pocket(features, lables, index_array, max_update_times, rate = 1):
    w_pocket = np.zeros((5, 1))
    w = np.zeros((5, 1))
    print pla_error_rate(features,lables,w)
    sample_len = len(features)
    flag = 1 ###algorithm halts flag, 1 for running
    index = 0
    count = 0
    while (flag):
        feature_index = index_array[index]
        ### check if need update w
        if lables[feature_index][0]*(np.dot(features[feature_index], w)[0]) <= 0:
            ### update w:  w = w + yi*xi  b = b + yi
            w = w + rate*lables[feature_index][0]*np.mat(features[feature_index]).T
            count += 1

            ### check if we need to update pocket
            if pla_error_rate(features, lables, w) < pla_error_rate(features, lables, w_pocket):
                print "-------------------"
                print w
                print pla_error_rate(features,lables,w)
                print w_pocket
                print pla_error_rate(features,lables,w_pocket)
                w_pocket = w

        if count >= max_update_times:
            flag = 0
        elif index >= sample_len - 1:
            index = 0
        else:
            index += 1
    return w_pocket,w

def pla_fix_index(features, lables, index_array, rate = 1):
    w = np.zeros((5, 1))
    sample_len = len(features)
    flag = 1 ###algorithm halts flag, 1 for running
    index = 0
    right_items = 0  ### if right_items == feature len, algorithm halts
    count = 0
    while (flag):
        feature_index = index_array[index]
        ### check if need update w
        if lables[feature_index][0]*(np.dot(features[feature_index], w)[0]) <= 0:
            ### update w:  w = w + yi*xi  b = b + yi
            w = w + rate*lables[feature_index][0]*np.mat(features[feature_index]).T
            ### clean right items
            right_items = 0
            count += 1
        else:
            ### update 
            right_items += 1

        if right_items >= sample_len:
            flag = 0
        elif index >= sample_len - 1:
            index = 0
        else:
            index += 1
    return count

### perceptron learning algorithm, input featrues and lables,learning rate, return w,number of iterations
def pla(features, lables, alpha = 1):
    w = np.zeros((5, 1))
    sample_len = len(features)
    flag = 1 ###algorithm halts flag, 1 for running
    index = 0
    right_items = 0  ### if right_items == feature len, algorithm halts
    count = 0
    while (flag):
        ### check if need update w
        if lables[index][0]*(np.dot(features[index], w)[0]) <= 0:
            ### update w:  w = w + yi*xi  b = b + yi
            w = w + lables[index][0]*np.mat(features[index]).T
            ### clean right items
            right_items = 0
            count += 1
        else:
            ### update 
            right_items += 1

        if right_items >= sample_len:
            flag = 0
        elif index >= sample_len - 1:
            index = 0
        else:
            index += 1
    return count
### import data from file
def load_data(file_path):
    print "in"
    file_object = open(file_path)
    try:
        lines = file_object.readlines()
    finally:
        file_object.close()

    sample_num = len(lines)

    x = np.zeros((sample_num, 5))
    y = np.zeros((sample_num, 1))

    index = 0
    for line in lines:
        ### split feature and label
        items = line.strip().split('\t')
        x[index][1:5] = np.array([float(num) for num in items[0].strip().split()])[:]
        x[index][0] = 1
        y[index][0] = float(items[1])
        index += 1
    print "indone"
    return x,y

if __name__ == '__main__':

    ### prolem 15
    """
    print "nihao"
    print "Q16"    
    (X,Y) = load_data('in')
    print pla(X,Y,rate)

    ### problem 16
    (X,Y) = load_data('in')
    print "Q16"    
    update_times_array = []
    for i in range(200):
        index_array = range(0,400)
        rd.shuffle(index_array)
        update_times = pla_fix_index(X, Y, index_array)
        update_times_array.append(update_times)
    print "Q16"    
    print np.mean(update_times_array)

    ### problem 17
    print "Q17"
    (X,Y) = load_data('in')
    update_times_array2 = []
    for i in range(2000):
        index_array = range(0,400)
        rd.shuffle(index_array)
        update_times = pla_fix_index(X, Y, index_array, rate=0.5)
        update_times_array2.append(update_times)
    print "Q17"
    print "Average num: ", sum(update_times_array2)/(len(update_times_array2)*1.0)
    """

    ### problem 18 and 19 and 20
    (X,Y) = load_data('train')
    (X_test,Y_test) = load_data('test')

    error_rate_array = []
    error_rate_array_50 = []
    w = np.zeros((5, 1))
    w_50 = np.zeros((5, 1))
    for i in range(1):
        print i
        index_array = range(0,len(X))
        rd.shuffle(index_array)
        ### train on the training set
        #(w,w_50) = pla_pocket(X, Y, index_array, 50)
        (w,w_100) = pla_pocket(X, Y, index_array, 100)

        ### test on the test set
        error_rate_array.append(pla_error_rate(X_test, Y_test, w))
        error_rate_array_50.append(pla_error_rate(X_test, Y_test, w_100))
        #error_rate_array.append(pla_error_rate(X_test, Y_test, w))

    print "average error rate on test set: ",sum(error_rate_array)/(len(error_rate_array)*1.0)
    print "average error rate on test set: ",sum(error_rate_array_50)/(len(error_rate_array)*1.0)
