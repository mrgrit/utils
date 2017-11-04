import queue
from threading import Thread

import os
import time
#import datetime
from datetime import date
from datetime import datetime
import pandas as pd
from pandas import HDFStore, DataFrame
import numpy
import xlrd

from hashtable import HashTable
import heapq
from graph import GraphSet, GraphDict
import utils
from collections import defaultdict, Counter

#import MySQLdb

from pprint import pprint

import h5py



#########################
ht_size = 0
HOUR_MIN = 0
SIP_DIP = 1
#########################

logdir = os.path.join('C:\\','users','mrgrit','desktop','ais','log',utils.get_logday())

#########################
dt_min_hs = HashTable(ht_size)
dt_hour_hs = HashTable(ht_size)
sip_hs = HashTable(ht_size)
dip_hs = HashTable(ht_size)
dport_hs = HashTable(ht_size)
pt_hs = HashTable(ht_size)

hour_min_hs = HashTable(100)
#########################



def add_hash(ht, dic):
    #dic_in = [(key, value) for key, value in dic.items()]
    for key, value in dic:
        ht.add(key, value)
    print(ht)
    


def graph_c(logtype, log):
    h5_path = os.path.join("d:\\","initiative","spade","h5")
    #file = h5py.File(h5_path,'r')
    event_number = 5
    if logtype == 'fw':
        pass
        
    elif logtype == 'ips' :
               
        #sd = createGraph(log,'sip','dip')
        #pprint(sd._graph)
        #sm = createGraph(log,'sip','method')
        #pprint(sm._graph)
        #md = createGraph(log,'method','dip')
        #pprint(md._graph)
        #ms = createGraph(log,'method','sip')
        #pprint(ms._graph)
        ms = createGraph(log,'method','sip')
        ms = {}

        store = pd.HDFStore(os.path.join(h5_path,"ms.h5"))
        store['ms'] = ms
        store['ms']
        pprint(ms._graph)

def h5_save(dfs,h5_name):
    h5_path = os.path.join("d:\\","initiative","spade","h5",h5_name)
    file = h5py.File(h5_path,'r')
    for df in dfs:
        dataset.file[df]
    #arr1ev = 
        
        

    

def createGraph(graph_df,node1,node2):
    default_list = []
    graphPayload = set()
    graphPayload = GraphSet(default_list, directed=True)    
    for graph_set in graph_df.loc[:,[node1,node2]].values:                
        graphPayload.add(graph_set[0],graph_set[1])      
    return graphPayload
    
    
            
def orderByValue(dic,size,direction):
        heap = [(-value, key) for key, value in dic.items()]
        largest = heapq.nsmallest(size,heap)        
        if direction == 'desc':
            largest = [(key, -value) for value, key in largest]
        
        
        return largest
    

def read_bulk():    
    today = date.today()
    for root, dirs, files in os.walk(logdir):
        for fname in files:
            logfile = os.path.join(root,fname)
            print(logfile)
            if 'ips' in fname:
                df = pd.read_excel(logfile,names=['logtype','time','_','sip','sport','dip','dport','result','method','not','_','_','_','_','att_code'])
                df["logtype"]="ips"
                df["write_date"]=today
                df["ans_flag"]=0
                log = df.loc[:,['logtype','time','sip','sport','dip','dport','result','method','not','att_code','write_date','ans_flag']]
                graph_c('ips',log)
                
            elif 'fw' in fname:                
                df = pd.read_excel(logfile,names=['logtype','time','_','_','sip','_','_','dip','dport','_','_','protocol','result','_','_','_'])
                df["logtype"]="fw"
                df["write_date"]=today                
                log = df.loc[:,['logtype','time','sip','dip','dport','protocol','result','write_date']]                
                #create_dict('fw',len(log),log)
                graph_c('fw',log)
                    
            elif 'waf' in fname:
                df = pd.read_excel(logfile,names=['logtype','time','_','_','_','sip','sport','dip','dport','_','_','result','method','_','_','_','_','gnp','att_code'])
                df["logtype"]="waf"
                df["write_date"]=today
                df["ans_flag"]=0
                log = df.loc[:,['logtype','time','sip','sport','dip','dport','result','method','gnp','att_code','write_date','ans_flag']]
                create_hash('waf',len(log),log)
                            
            elif 'web' in fname:                
                df = pd.read_excel(logfile,names=['logtype','time','_','_','_','sip','_','dip','_','_','_','result','method','_','_','_','_','gnp','att_code'])
                df["logtype"]="web"
                df["write_date"]=today
                df["ans_flag"]=0
                log = df.loc[:,['logtype','time','sip','dip','result','method','gnp','att_code','write_date','ans_flag']]
                create_hash('web',len(log),log)

def loadHashTables(table):
    if table == 'all':
        # 국가번호 HT
        #select cc,
        pass
    


if __name__ == '__main__':
    #conn = MySQLdb.connect('localhost','root','rhrnak#33','graph', charset = "utf8")
    loadHashTables('all')
    read_bulk()
    #print(dt_hour_hs)






