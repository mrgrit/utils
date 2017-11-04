import queue
from threading import Thread

import os
import time
#import datetime
from datetime import date
from datetime import datetime
import pandas as pd
import numpy
import xlrd

from hashtable import HashTable
import heapq
from graph import GraphSet, GraphDict
import utils


#import MySQLdb

from pprint import pprint


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
    
def create_dict(logtype, size, log):
    dt_min_dict = dict()
    dt_hour_dict = dict()
    sip_dict = dict()
    dip_dict = dict()

    if logtype == 'fw':        
        for date in log['time']:
            # 시간 집계
            date_min = date[0:-3]
            cnt = dt_min_dict.get(date_min,0)
            dt_min_dict[date_min]=cnt+1
            # 분 집계
            date_hour = date[0:-6]
            cnt = dt_hour_dict.get(date_hour,0)
            dt_hour_dict[date_hour]=cnt+1
            
        dt_min_dict = orderByValue(dt_min_dict,len(dt_min_dict),'desc')        
        dt_hour_dict = orderByValue(dt_hour_dict,len(dt_hour_dict),'desc')        
        hour_min = [dt_hour_dict,dt_min_dict]        
        create_graph(HOUR_MIN,hour_min)

        # just 횟수
        for sip in log['sip']:
            # sip 집계 
            cnt = sip_dict.get(sip,0)
            sip_dict[sip] = cnt+1
        for dip in log['dip']:
            cnt = dip_dict.get(dip,0)
            dip_dict[dip] = cnt+1

        sip_dict = orderByValue(sip_dict,len(sip_dict),'desc')        
        dip_dict = orderByValue(dip_dict,len(dip_dict),'desc')
        # graph를 위해서 ipset이 필요함
        ipset=log.iloc[:,[2,3]]
        sip_dip = [sip_dict, dip_dict, ipset]
        create_graph(SIP_DIP, sip_dip)

        # ipset['sip'] => sip, ipset['dip'] => dip
        # {sip:[dip,time]}        
        #ipset_dict = dict()
        #ip_count_dict = dict()        
        #for sip in sip_dict:
        #    dip_li = list()
        #    for sip_t in ipset['sip']:                
        #        if sip[0] == sip_t:
        #            dip_li.append(ipset['dip'])
        #    ipset_dict[sip[0]]=dip_li
            #print(len(ipset_dict[sip]))
        #    ip_count_dict[sip[0]] = len(ipset_dict[sip[0]])
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(ip_count_dict)
        #ip_count_dict = orderByValue(ip_count_dict,len(ip_count_dict),'desc')
    
    

def create_graph(graph_type, dics):
    if graph_type == HOUR_MIN :               
        # dics[0] => hour, dics[1] => min
        hour_min = dict()        
        for hour_key, hour_value in dics[0] :
            hour_min_li = list()
            for min_key, min_value in dics[1]:                               
                if hour_key in min_key:                    
                    hour_min_li.append({min_key:min_value})
            hour_min[hour_key] = hour_min_li

        pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(hour_min['2017-09-13 00'])

    elif graph_type == SIP_DIP :
        # dics[0] => sip, dics[1] => dip, dics[2] => ipset
        # ipset['sip'] => sip, ipset['dip'] => dip
        # {sip:[dip,time]}        
        ipset_dict = dict()
        ip_count_dict = dict()        
        for sip in dics[0]:
            
            dip_set = set()
            for i, sip_t in enumerate(dics[2]['sip']):
                #print(sip_t)
                if sip[0] == sip_t:
                    #print(dics[2]['dip'][i])
                    dip_set=dics[2]['dip'][i]
                    #print(dip_set)
            ipset_dict[sip[0]]=dip_set
            #print(ipset_dict)
            #print(len(ipset_dict[sip]))
            ip_count_dict[sip[0]] = len(ipset_dict[sip[0]])
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(ip_count_dict)
        ip_count_dict = orderByValue(ip_count_dict,len(ip_count_dict),'desc')
        print(ip_count_dict)

def graph_c(logtype, log):
    dt_min_dict = dict()
    dt_hour_dict = dict()
    sip_dict = dict()
    dip_dict = dict()
    mh = dict()

    fw_logs = []
    ips_logs = []

    if logtype == 'fw':
        fwghm = GraphDict(fw_logs, directed=True)
        #fwgmc = Graph(mc_logs, directed=True)
        for date in log['time']:
            # 시간 집계
            date_hour = date[0:-6]            
            # 분 집계            
            date_min = date[0:-3]
            cnt = dt_min_dict.get(date_min,0)
            dt_min_dict[date_min]=cnt+1
            #fwghm.add(date_hour, 

            if date_hour in date_min:
                #print(dt_min_dict)
                #fwghm.add(date_hour, orderByValue(dt_min_dict[date_min], len(dt_min_dict),'desc'))
                fwghm.add(date_hour, {date_min:cnt})
                #mh[date_hour] = orderByValue(dt_min_dict, len(dt_min_dict),'desc')
                #dt_min_dict={}
                
    elif logtype == 'ips' :
        ips_sd = set()
        mem = born_graph(ips_sd,log,'sip','dip')
        pprint(mem._graph)    

    

def born_graph(graph_mem,graph_df,node1,node2):
    default_list = []
    graph_mem = GraphSet(default_list, directed=True)

    for graph_set in graph_df.loc[:,[node1,node2]].values:
        graph_mem.add(graph_set[0],graph_set[1])

    return graph_mem
    
    
            
           
            
        
            
            
        

        #pprint(ipsgsd._graph)
        
            
            
            
    #pprint(len(ipsgsd._graph))
    #pprint(ipsgsm._graph)
    #pprint(mh)
                

            

            
                        
        

        

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






