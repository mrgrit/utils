import MySQLdb
import pandas as pd
import numpy as np
from datetime import date

import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


###############   CONST   #############
PCA_DIM = 10
LEARNING_START_DATE = '2017-06-01'
LEARNING_END_DATE = date.today()
#######################################



# PCA
from sklearn.decomposition import PCA
def do_pca(dataset,n):
    pca = PCA(n_components=n)
    pca.fit(dataset)
    print(pca.explained_variance_ratio_)
    x_train_transformed = pca.fit_transform(dataset)
    #print(x_train_transformed)
    return x_train_transformed


def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion Matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize :
        cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
    else:
        print("confusion Matrix, without normalization")
    
    print(cm)
    
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('predicted label')
    plt.show()
    
def do_confusion(test,pred):

    cnf_matrix = confusion_matrix(test,pred)
    print(cnf_matrix.shape)
    # bl을 bl로 맞춘 정확도
    accuracy = cnf_matrix[1][1]/(cnf_matrix[1][0]+cnf_matrix[1][1])
    
    plot_confusion_matrix(cnf_matrix, classes=['wl','bl'])
    
    #return accuracy_score(test, pred)
    return accuracy

def do_f1(test, pred):
    return f1_score(test, pred)

from sklearn import tree

def do_decision(x_train, y_train, test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(test)
    
    return y_pred

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

def do_svm(x,y,test,c,g):
    clf = SVC(C=c, kernel='rbf',gamma=g) 
    print("fit => ", clf.fit(x,y))
    y_pred = clf.predict(test)    
    return y_pred

def tune_svm(x,y,test):
    print('x => ', x)
    print('y => ',len(y==0), y)
    print('test => ', test)
    tuned_parameters = [{'kernel':['rbf'], 
                         'gamma':[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9], 
                         'C' : [10,100,1000,10000,100000,1000000,10000000]}]
    
    scores = ['precision','recall']
    
    for score in scores:
        print("# Tunning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(x,y)
        
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print("Grid scores on development set:")
        print()
        
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std *2, params))

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full deveopment set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(test)
        print(classification_report(y_true, y_pred))
        print()


conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")

# db에서 학습 할 데이터 불러와서 전처리
def data_preprocessing(start_date, end_date):
    curs = conn.cursor()
    data_s = 'select ip, bl_ibm, times, web_ok, web_rej, fw_pd, fw_db, waf_ok, waf_rej, ips_pd, ips_db, bl_dg_yn from aw_ip_cache where write_date >= %s and write_date <=%s and (ip_gubun = 1 or ip_gubun = 2) and bl_ibm is not null'
    curs.execute(data_s,(start_date,end_date))
    rows = curs.fetchall()
    
    df = pd.DataFrame(list(rows), columns=['ip', 'bl_ibm', 'times', 'web_ok', 'web_rej', 'fw_pd', 'fw_db', 'waf_ok', 'waf_rej', 'ips_pd', 'ips_db',' bl_dg_yn'])
    
    df['bl_dg_yn'] = 0
    bl = pd.read_csv('bl.csv',names=['ip'])
    

    # bl.csv 에 기록된 black list를 읽어 실제 차단한 ip를 기록
    for bl_ip in bl['ip'] :
        df.ix[df['ip']==bl_ip,'bl_dg_yn'] = 1        

    print("total bl count : ",len(df.loc[df['bl_dg_yn']==1]))
    
    y_label = df['bl_dg_yn']

    # axis = 1 : row
    # pca에 불필요한 column은 삭제
    df = df.drop('bl_dg_yn',1)
    df = df.drop('ip',1)

    # 10차원 -> PCA_DIM 으로 축소
    print(df.head())
    df = do_pca(df,PCA_DIM)

    df = np.column_stack((df,y_label))

    wl = df[y_label == 0]
    bl = df[y_label == 1]

    print("wl = ",len(wl))
    print("bl = ",len(bl))

    x_train_w, x_test_w = np.split(wl,[int(.8*len(wl))])
    x_train_b, x_test_b = np.split(bl,[int(.8*len(bl))])

    x_train = np.concatenate([x_train_w, x_train_b])
    x_test = np.concatenate([x_test_w, x_test_b])

    y_train = x_train[:,PCA_DIM]
    y_test = x_test[:,PCA_DIM]

    x_train = x_train[:,0:PCA_DIM]
    x_test = x_test[:,0:PCA_DIM]
    print(x_train.shape)
    print(x_test.shape)
    
    #tune_svm(x_train, y_train, x_test)
    y_pred = do_svm(x_train, y_train, x_test, 1000000, 1e-06)
    #print(y_pred[1])
    print("SVM Accuracy: ", do_confusion(y_test, y_pred))
    print("f1 score: ", do_f1(y_test, y_pred))

    #y_pred = do_decision(x_train, y_train, x_test)
    
    #print("DT accuracy: ", do_confusion(y_test, y_pred))
    #print("F1 Score: ", do_f1(y_test, y_pred))

data_preprocessing(LEARNING_START_DATE,LEARNING_END_DATE)

conn.close()
                     

