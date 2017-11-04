import queue
from threading import Thread

import json
from datetime import date
import os
import time
import datetime
from datetime import date
import pandas as pd
import xlrd
from elasticsearch import Elasticsearch
import pycurl
import requests

max_result_window = 5000000

es = Elasticsearch([{'host': '10.24.50.20', 'port': 9200}])

logq = queue.Queue()
day = datetime.date.today()

def get_logday() :
    logday_list = []
    logday = ''
    today = str(datetime.date.today())
    today = today.split('-')        
    logday = str(today[0]) + str(today[1])+str(today[2])        
    return logday

def write_log(msg) :    
    #log_file = logdir+str(logday)+logfile
    now = datetime.datetime.now()
    day = datetime.date.today()
    f = open('./'+str(day)+'_log.txt','a')
    log_message = '['+ str(now) + ']  ' + msg + '\n'
    f.write(log_message)
    f.close()

logdir = os.path.join('H:\\tf10','log',get_logday())

def read_logfull():    
    for root, dirs, files in os.walk(logdir):
        for fname in files:
            logfile = os.path.join(root,fname)
            print(logfile)
            if 'ips' in fname:
                df = pd.read_excel(logfile)
                df["B/S"]="ips"
                for i in range(len(df)):                    
                    ips = df.loc[[i],['B/S','사건일시','Source IP','Source Port','Destination IP','Destination Port','결과','Method','횟수','비고']]   #9                    
                    logq.put_nowait(ips)
            elif 'fw' in fname:
                df = pd.read_excel(logfile)
                df["B/S"]="fw"
                for i in range(len(df)):                    
                    fw = df.loc[[i],['B/S','수집일시','Source IP','Destination IP','Destination Port','Protocol','결과']]    #6    
                    logq.put_nowait(fw)
            elif 'waf' in fname:
                df = pd.read_excel(logfile)
                df["B/S"]="waf"
                for i in range(len(df)):                    
                    waf = df.loc[[i],['B/S','사건일시','Source IP','Source Port','Destination IP','Destination Port','결과','Method','Get/Post','비고']]        
                    logq.put_nowait(waf)
            elif 'web' in fname:                
                df = pd.read_excel(logfile)
                df["B/S"]="web"
                for i in range(len(df)):       
                    web = df.loc[[i],['B/S','사건일시','Source IP','Destination IP','결과','Method','비고']]        
                    logq.put_nowait(web)

    
def write_logfull() :    
    while True:
        if logq.qsize() == 0 :
            time.sleep(2)
            print("sleeping")
            write_log("sleeping")
        else:            
            for i in range(logq.qsize()):
                log=logq.get_nowait()
                if logq.qsize() % 200 == 0 :
                    print(logq.qsize())
                    qsiz = "Queue Size = " + str(logq.qsize())
                    write_log(qsiz)
                if log.iloc[0,0] == 'ips' :
                    print(log.iloc[0,0])
                    ips_json = {'logtype':log.iloc[0,0],'time':log.iloc[0,1],'sip':log.iloc[0,2],'sport':str(log.iloc[0,3]),'dip':log.iloc[0,4],'dport':str(log.iloc[0,5]),'result':log.iloc[0,6],'method':log.iloc[0,7],'not':str(log.iloc[0,8]),'att_code':log.iloc[0,9]}
                    url = 'http://10.24.50.20:9200/ais/logfull/'
                    payload = ips_json
                    headers = {'Content-type':'application/json'}
                    r = requests.post(url, data = json.dumps(payload), headers=headers)                    
                elif log.iloc[0,0] == 'fw' :                    
                    fw_json = {'logtype':log.iloc[0,0],'time':log.iloc[0,1],'sip':log.iloc[0,2],'dip':log.iloc[0,3],'dport':str(log.iloc[0,4]),'protocol':log.iloc[0,5],'result':log.iloc[0,6]}
                    url = 'http://10.24.50.20:9200/ais/logfull/'
                    payload = fw_json
                    headers = {'Content-type':'application/json'}
                    r = requests.post(url, data = json.dumps(payload), headers=headers)
                elif log.iloc[0,0] == 'waf' :                    
                    waf_json = {'logtype':log.iloc[0,0],'time':log.iloc[0,1],'sip':log.iloc[0,2],'sport':str(log.iloc[0,3]),'dip':log.iloc[0,4],'dport':str(log.iloc[0,5]),'result':log.iloc[0,6],'method':log.iloc[0,7],'gnp':log.iloc[0,8],'att_code':log.iloc[0,9]}
                    url = 'http://10.24.50.20:9200/ais/logfull/'
                    payload = waf_json
                    headers = {'Content-type':'application/json'}
                    r = requests.post(url, data = json.dumps(payload), headers=headers)                    
                elif log.iloc[0,0] == 'web' :                    
                    web_json = {'logtype':log.iloc[0,0],'time':log.iloc[0,1],'sip':log.iloc[0,2],'dip':log.iloc[0,3],'result':log.iloc[0,4],'method':log.iloc[0,5],'att_code':log.iloc[0,6]}
                    url = 'http://10.24.50.20:9200/ais/logfull/'
                    payload = web_json
                    headers = {'Content-type':'application/json'}
                    r = requests.post(url, data = json.dumps(payload), headers=headers)
                    

def main():
    write_log("start")
    read_th = Thread(target=read_logfull)    
    write_th = Thread(target=write_logfull)    
    #read_th.demon = True
    #write_th.demon = True
    read_th.start()    
    write_th.start()
    
if __name__ == '__main__':
    main()

