#-- coding:utf-8 --
import random as rd
f = open("train","r");
x=[];
y=[];
for line in f.readlines():
    line = line.strip();
    listx = ['1']
    listx.extend(line.split("\t")[0].split(" "))
    x.append([float(item) for item in listx]);
    y.append(int(line.split("\t")[1]));
fx = open("test","r");
testx=[];
testy=[];
for line in fx.readlines():
    line = line.strip();
    listx = ['1']
    listx.extend(line.split("\t")[0].split(" "))
    testx.append([float(item) for item in listx]);
    testy.append(int(line.split("\t")[1]));
def cal(item, w,yf):
    ret=0;
    for i in range(0,5):
        ret = ret+item[i]*w[i];
    ret = ret*yf
    if (ret>0): return 1;
    else: return -1
def fix(item,w,yf,rate=1.0):
    wnew = w[:];
    for i in range(0,5):
        wnew[i]=w[i]+yf * item[i]*rate;
    return wnew;
def testw(xx,yy,ww):
    cnt = 0;
    for i in range(0,len(yy)):
        tmpy = cal(xx[i],ww,yy[i]);
        if (tmpy <0):
            cnt = cnt+1;
    return cnt*1.0/len(yy);

def pocket(x,y,arrayx,printans=True,rate=1.0,update_cnt=50,force_update=False):
    w=[0.0,0.0,0.0,0.0,0.0]
    wfix=[0.0,0.0,0.0,0.0,0.0]
    error_rate = 1.0;
    fixcnt=0;
    while 1:
        fixed=0;
        for i in arrayx:
            item=x[i]
            tmpy = cal(item,wfix,y[i]);
            if (tmpy<0):
                fixcnt=fixcnt+1;
                fixed=1;
                wfix = fix(item,wfix,y[i],rate);
                error_rate_new = testw(x,y,wfix);
                if (error_rate_new < error_rate or force_update):
                    w = wfix[:]
                    error_rate = error_rate_new
                if (fixcnt>=update_cnt):
                    return testw(testx,testy,w);
        if (fixed==0): 
            break;
    if (printans):
        print w;
        print fixcnt
        print error_rate
    return testw(testx,testy,w);
def PLA(x,y,arrayx,printans=True,rate=1.0):
    w=[0.0,0.0,0.0,0.0,0.0]
    fixcnt=0;
    while 1:
        fixed=0;
        for i in arrayx:
            item=x[i]
            tmpy = cal(item,w);
            if (tmpy<=0):
                fixcnt=fixcnt+1;
                fixed=1;
                fix(item,w,y[i],rate);
        if (fixed==0): 
            break;
    if (printans):
        print w;
        print fixcnt
    return fixcnt    
print "Q18"
print pocket(x,y,range(0,len(y)));
print "Q18"
sum = 0;
for i in range(0,200):
    if (i%100==0):
        print i
    index_array = range(0,400)
    rd.shuffle(index_array)
    sum = sum+pocket(x,y,index_array,0);
print sum/200.0
print "Q19"
sum = 0;
for i in range(0,200):
    index_array = range(0,400)
    rd.shuffle(index_array)
    sum = sum+pocket(x,y,index_array,0,force_update=True);
print sum/200.0

print "Q20"
sum = 0;
for i in range(0,200):
    if (i%100==0):
        print i
    index_array = range(0,400)
    rd.shuffle(index_array)
    sum = sum+pocket(x,y,index_array,0,update_cnt=100);
print sum/200.0
print "Q19"
sum = 0;
for i in range(0,200):
    index_array = range(0,400)
    rd.shuffle(index_array)
    sum = sum+pocket(x,y,index_array,0,force_update=True,update_cnt=100);
print sum/200.0

