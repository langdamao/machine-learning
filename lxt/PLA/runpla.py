# -- coding:utf-8 --
import random as rd
def cal(item, w):
    ret=0;
    for i in range(0,5):
        ret = ret+item[i]*w[i];
    if (ret>0): return 1;
    else: return -1
def fix(item,w,yf,rate=1.0):
    for i in range(0,5):
        w[i]=w[i]+yf * item[i]*rate;
def PLA(x,y,arrayx,printans=True,rate=1.0):
    w=[0.0,0.0,0.0,0.0,0.0]
    fixcnt=0;
    while 1:
        fixed=0;
        for i in arrayx:
            item=x[i]
            tmpy = cal(item,w);
            if (tmpy!=y[i]):
                fixcnt=fixcnt+1;
                fixed=1;
                fix(item,w,y[i],rate);
        if (fixed==0): 
            break;
    if (printans):
        print w;
        print fixcnt
    return fixcnt    
f = open("in","r");
x=[];
y=[];
for line in f.readlines():
    line = line.strip();
    listx = ['1']
    listx.extend(line.split("\t")[0].split(" "))
    x.append([float(item) for item in listx]);
    y.append(int(line.split("\t")[1]));
print "Q15"
PLA(x,y,range(0,len(y)));
print "Q16"
sum = 0;
for i in range(0,2000):
    index_array = range(0,400)
    rd.shuffle(index_array)
    sum = sum+PLA(x,y,index_array,0);
print sum/2000.0
print "Q17"
sum = 0;
for i in range(0,2000):
    index_array = range(0,400)
    rd.shuffle(index_array)
    sum = sum+PLA(x,y,index_array,0,0.5);
print sum/2000.0

