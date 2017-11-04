import queue
from threading import Thread

import json # is not JSON serializable 에러 해결 방법 -_-
from datetime import date
import os
import time
import datetime
from datetime import date
import pandas as pd
import numpy
import xlrd
from elasticsearch import Elasticsearch
import pycurl
import requests

#######CONST#########
THREAD_ONOFF = "OFF"

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

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.int64):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

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
                    ips = df.loc[[i],['B/S','사건일시','Source IP','Source Port','Destination IP','Destination Port','결과','Method','횟수','비고']]
                    logq.put_nowait(ips)
            elif 'fw' in fname:
                df = pd.read_excel(logfile)
                df["B/S"]="fw"
                for i in range(len(df)):                    
                    fw = df.loc[[i],['B/S','수집일시','Source IP','Destination IP','Destination Port','Protocol','결과']]
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
                    
def read_logfull_bulk():
    tmp_dir = os.path.join("h:\\","tf10","elk","tmp")
    today = datetime.date.today()
    for root, dirs, files in os.walk(logdir):
        for fname in files:
            logfile = os.path.join(root,fname)
            print(logfile)
            if 'ips' in fname:
                df = pd.read_excel(logfile,names=['logtype','time','_','sip','sport','dip','dport','result','method','not','_','_','_','_','att_code'])
                df["logtype"]="ips"
                df["write_date"]=today
                log = df.loc[:,['logtype','time','sip','sport','dip','dport','result','method','not','att_code']]
                with open('temp.json','w') as f:                    
                    f.write(log.to_json(orient='records', lines=True))
                with open('temp.json','r') as f:                    
                    json_path = os.path.join(tmp_dir,str(time.time()))
                    with open(json_path,'w') as g:
                        while True:
                            line = f.readline()
                            if not line :                                
                                break
                            g.write("{\"index\":{} }\n")
                            g.write(line)
                #url = 'http://10.24.50.20:9200/ais/logfull/_bulk'
                #payload = "--date-binary "+"@"+json_path
                #headers = {'Content-type':'application/json'}
                #r = requests.post(url, data = json.dumps(payload), headers=headers)
                es.bulk("ais","logfull",json_path)
                os.remode(json_path)
                return
            elif 'fw' in fname:
                df = pd.read_excel(logfile)
                df["B/S"]="fw"
                for i in range(len(df)):                    
                    fw = df.loc[[i],['B/S','수집일시','Source IP','Destination IP','Destination Port','Protocol','결과']]
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
                    
def elastic_bulk():
    for root, dirs, files in os.walk(logdir):
        for fname in files:
            logfile = os.path.join(root,fname)
            print(logfile)
            if 'ips' in fname:
                df = pd.read_excel(logfile)                             
                log = df.loc[:,['사건일시','Source IP','Source Port','Destination IP','Destination Port','결과','Method','횟수','비고']]        
                for i in range(len(log)):
                    ips_json = {'logtype':'ips','time':log.iloc[i,0],'sip':log.iloc[i,1],'sport':str(log.iloc[i,2]),'dip':log.iloc[i,3],'dport':str(log.iloc[i,4]),'result':log.iloc[i,5],'method':log.iloc[i,6],'not':str(log.iloc[i,7]),'att_code':log.iloc[i,8]}
                    write_log(str(ips_json))
                    url = 'http://10.24.50.20:9200/ais/logfull/'
                    payload = ips_json
                    headers = {'Content-type':'application/json'}
                    r = requests.post(url, data = json.dumps(payload), headers=headers)
                    print(r)
            elif 'fw' in fname:
                df = pd.read_excel(logfile)
                log = df.loc[:,['수집일시','Source IP','Destination IP','Destination Port','Protocol','결과']]                            
                fw_json = {'logtype':'fw','time':log.iloc[0,0],'sip':log.iloc[0,1],'dip':log.iloc[0,2],'dport':str(log.iloc[0,3]),'protocol':log.iloc[0,4],'result':log.iloc[0,5]}
                url = 'http://10.24.50.20:9200/ais/logfull/_bulk'
                payload = fw_json
                headers = {'Content-type':'application/json'}
                r = requests.post(url, data = json.dumps(payload), headers=headers)
            elif 'waf' in fname:
                df = pd.read_excel(logfile)                
                log = df.loc[[i],['사건일시','Source IP','Source Port','Destination IP','Destination Port','결과','Method','Get/Post','비고']]                            
                waf_json = {'logtype':'waf','time':log.iloc[0,0],'sip':log.iloc[0,1],'sport':str(log.iloc[0,2]),'dip':log.iloc[0,3],'dport':str(log.iloc[0,4]),'result':log.iloc[0,5],'method':log.iloc[0,6],'gnp':log.iloc[0,7],'att_code':log.iloc[0,8]}
                url = 'http://10.24.50.20:9200/ais/logfull/_bulk'
                payload = waf_json
                headers = {'Content-type':'application/json'}
            elif 'web' in fname:                
                df = pd.read_excel(logfile)
                log = df.loc[:,['사건일시','Source IP','Destination IP','결과','Method','비고']]
                log.to_json(orient='index')
                print(log)
                for i in range(len(log)):
                    web_json = {'logtype':'web','time':log.iloc[i,0],'sip':log.iloc[i,1],'dip':log.iloc[i,2],'result':log.iloc[i,3],'method':log.iloc[i,4],'att_code':log.iloc[i,5]}
                    url = 'http://10.24.50.20:9200/ais/logfull/'
                    payload = web_json
                    headers = {'Content-type':'application/json'}
                    r = requests.post(url, data = json.dumps(payload), headers=headers)
                    if i % 200 == 0 : print("web =>" ,i)
                    
    
                    

def main():
    write_log("start")
    if THREAD_ONOFF == "ON" :
        read_th = Thread(target=read_logfull)    
        write_th = Thread(target=write_logfull)        
        read_th.start()    
        write_th.start()
    elif THREAD_ONOFF == "OFF":
        read_logfull_bulk()
        return
    
if __name__ == '__main__':
    main()

