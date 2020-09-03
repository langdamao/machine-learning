import numpy as np
from sklearn.svm import SVC
X = [[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]
Y = [-1,-1,-1,1,1,1,1]
clf = SVC(C=1e10,kernel='poly',degree=2,gamma=1,coef0=1)
clf.fit(X,Y);
print(clf.dual_coef_)
print(np.sum(clf.dual_coef_))
print(clf.support_)
print(clf.support_vectors_)
#print(clf.coef_)
print(clf.intercept_)
print(clf.n_support_)
print(clf.predict(X))
X = [[1,1],[1,-1],[-1,-1],[-1,1],[1,2],[-1,-2],[-2,1]]
print(clf.predict(X))
ww = [[0,0,2,1],[0,0,-2,1],[-2,1,0,0],[0,0,4,4],[0,0,-4,4]]
al = clf.dual_coef_.reshape(1,5)
w = list(np.dot(al,ww).reshape(4))
#print(w)
w.append(clf.intercept_[0])
print(w)
print(9*np.array(w))







