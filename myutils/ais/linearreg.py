#select distinct(method) from aw_log_full

# -*- coding: utf-8 -*-

import os
#from datetime import datetime
import datetime
from datetime import date
import MySQLdb


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier




path = "D:\\ais\\ai\log\old\sys"

conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")
sel = 'select count(*),count(distinct source_ip) from aw_log_full where write_date = %s'

curs = conn.cursor()

def do_rf(x, y, test):
    # X : 변수
    # y : 결과
    # test : test할 변수
    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(x, y)    
    y_pred = clf.predict(test)
    print(y_pred)
    return y_pred


def do_nn(x,y,test,layers,a):
    clf = MLPClassifier(solver='lbfgs',alpha=a,hidden_layer_sizes=(2,layers),random_state=1)
    clf.fit(x,y)
    y_pred = clf.predict(test)
    return y_pred

def do_svm(x,y,test,c,g):    
    clf = SVC(C=c, kernel='rbf',gamma=g) 
    print("fit => ", clf.fit(x,y))
    y_pred = clf.predict(test)    
    return y_pred

from sklearn import tree

def do_dt(x, y, test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x, y)
    y_pred = clf.predict(test)    
    return y_pred


def pre_proc():
    for root, dirs, files in os.walk(path):
        for fname in files:
            #fname => file name
            print('fname=>',fname)
            full_fname = os.path.join(root,fname)

            print(full_fname)
            f = open(full_fname,'r')
            line = f.readlines()
            f2 = open('result.txt','a')
            curs.execute(sel,(fname.strip('_log.txt'),))
            log_row = curs.fetchone()        

            start = line[0].strip('\n')
            end = line[len(line)-1].strip('\n')
 
            start = str(start.split('.')).split(' ')      
            start = start[1].strip("',")
        
            end = str(end.split('.')).split(' ')        
            end = end[1].strip("',")

            start = datetime.strptime(start,'%H:%M:%S')
            end = datetime.strptime(end,'%H:%M:%S')

            second = (end-start).seconds
        
            content = str(second)+'|'+str(log_row[0])+'|'+str(log_row[1])+'\n'
            f2.write(content)

            f.close
            f2.close

#pre_proc()
f = open('result.txt','r')
lines = f.readlines()
time = []
logs = []
ips = []
x = []
y=[]
test = []
for line in lines:
    line = line.split('|')    
    x=[int(line[1]),int(line[2].strip('\n'))] 
    logs.append(x)
    
    time.append(int(line[0]))

print(logs)
print(time)
tod = str(datetime.date.today())
curs.execute(sel,(tod,))
today = curs.fetchone()
print(today)
test = today
pred = do_rf(logs,time,test)
print(int(pred/60)," 분 후에 완료될 예정입니다(rf).")
pred = do_svm(logs,time,test,1000000, 0.00001)
print(int(pred/60)," 분 후에 완료될 예정입니다(svm).")
pred = do_nn(logs,time,test,3, 1e-12)
print(int(pred/60)," 분 후에 완료될 예정입니다(nn).")
pred = do_dt(logs,time,test)
print(int(pred/60)," 분 후에 완료될 예정입니다(dt).")

#X = zip(logs,ips)
#print(X)
#print(time)

#231분 vs 248분
# 13:27 vs 13:44

# from 09:36





        




        





