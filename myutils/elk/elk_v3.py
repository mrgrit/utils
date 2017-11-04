import queue
from threading import Thread

import json
from datetime import date
import os
import time
import datetime
from datetime import date
import pandas as pd
import numpy
import xlrd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pycurl
import requests

#######CONST#########
THREAD_ONOFF = "OFF"

es = Elasticsearch([{'host': '10.24.50.20', 'port': 9200}])

logq = queue.Queue()
fwq = queue.Queue()
ipsq = queue.Queue()
wafq = queue.Queue()
webq = queue.Queue()

ipq = queue.Queue()

day = datetime.date.today()


def get_logday() :
    logday_list = []
    logday = ''
    today = str(datetime.date.today())
    today = today.split('-')        
    logday = str(today[0]) + str(today[1])+str(today[2])        
    return logday

logdir = os.path.join('H:\\tf10','log',get_logday())

def write_log(msg) :    
    #log_file = logdir+str(logday)+logfile
    now = datetime.datetime.now()
    day = datetime.date.today()
    f = open('./'+str(day)+'_log.txt','a')
    log_message = '['+ str(now) + ']  ' + msg + '\n'
    f.write(log_message)
    f.close()

def file_handling(logtype,log):
    tmp_dir = os.path.join("h:\\","tf10","elk","tmp")
    with open(logtype,'w') as f:
        f.write(log.to_json(orient='records', lines=True))        
        with open(logtype,'r') as f:                    
            json_path = os.path.join(tmp_dir,str(time.time()))
            with open(json_path,'w') as g:
                while True:
                    line = f.readline()
                    if not line :                                
                        break
                    g.write("{\"index\":{}}\n")
                    g.write(line)
    return
                            
def read_bulk():    
    today = datetime.date.today()
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
                file_handling('ips',log)
                
            elif 'fw' in fname:                
                df = pd.read_excel(logfile,names=['logtype','time','_','_','sip','sport','_','dip','dport','_','_','protocol','result','_','_','_'])
                df["logtype"]="fw"
                df["write_date"]=today
                df["ans_flag"]=0
                log = df.loc[:,['logtype','time','sip','sport','dip','dport','protocol','result','write_date','ans_flag']]
                file_handling('fw',log)
                    
            elif 'waf' in fname:
                df = pd.read_excel(logfile,names=['logtype','time','_','_','_','sip','sport','dip','dport','_','_','result','method','_','_','_','_','gnp','att_code'])
                df["logtype"]="waf"
                df["write_date"]=today
                df["ans_flag"]=0
                log = df.loc[:,['logtype','time','sip','sport','dip','dport','result','method','gnp','att_code','write_date','ans_flag']]
                file_handling('waf',log)
                            
            elif 'web' in fname:                
                df = pd.read_excel(logfile,names=['logtype','time','_','_','_','sip','_','dip','_','_','_','result','method','_','_','_','_','gnp','att_code'])
                df["logtype"]="web"
                df["write_date"]=today
                df["ans_flag"]=0
                log = df.loc[:,['logtype','time','sip','dip','result','method','gnp','att_code','write_date','ans_flag']]
                file_handling('web',log)

def write_bulk():
    json_dir = os.path.join('H:\\tf10','elk','tmp')
    for root, dirs, files in os.walk(json_dir):            
        for fname in files:
            json_file = os.path.join(root,fname)            
            curl = "curl -s -XPOST \"10.24.50.20:9200/ais/logfull/_bulk\" -H \"Content-Type: application/json\" --data-binary @"+json_file
            write_log(curl)
            os.system(curl)
            os.remove(json_file)


#def q_push():
#    while True:
#        XGET '10.24.50.20:9200/ais/logfull
    
    

                    
def main():
    write_log("start")
    if THREAD_ONOFF == "ON" :
        read_th = Thread(target=read_bulk)    
        write_th = Thread(target=write_bulk)        
        read_th.start()    
        write_th.start()
    elif THREAD_ONOFF == "OFF":
        read_bulk()
        write_bulk()
        
        return
    
if __name__ == '__main__':
    main()

