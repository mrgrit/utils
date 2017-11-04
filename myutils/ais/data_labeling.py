import requests
import json
import pycurl
from io import BytesIO
import certifi
import datetime
import xlrd
import MySQLdb
import shutil
import os
import time

conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")
curs = conn.cursor()

def ipr_basic_susp(ip) :   # KR 일 때 건너뛰는 부분 제거 : 사설망 빼고 IP 넣으면 무조건 IBM 검사
    curs = conn.cursor()
    now = str(datetime.datetime.now())
    ipr_li = []
    ipr = ''
    ip3 = ip.split('.')
    tot = 0    
    print("inspecting bl_ibm : ",ip)
    if ip3[0] == '10' or (ip3[0] == '118' and ip3[1] == '219') or (ip3[0] == '192' and ip3[1] == '168') or (ip3[0] == '172' and ip3[1] == '16') or (ip3[0] == '172' and ip3[1] == '17') or (ip3[0] == '172' and ip3[1] == '30'):
        ipr_li = ['INT',0]
        return 
    else : 
        try :    
            res = requests.get('http://geoip.nekudo.com/api/'+ip)
            rej = res.json()
        except :
            ipr_li = ['ERR',0]
            #return ipr_li
            u_bl_ibm = 'update aw_ip_cache set cc=%s, bl_ibm=%s where ip = %s'
            curs.execute(u_bl_ibm,(ipr_li[0],ipr_li[1],ip))
            conn.commit()
            return
        try:
            if rej['country'] == [] :
                ipr_li = ['ERR', 0]
                u_bl_ibm = 'update aw_ip_cache set cc=%s, bl_ibm=%s where ip = %s'
                curs.execute(u_bl_ibm,(ipr_li[0],ipr_li[1],ip))
                conn.commit()
                return
            else :
                buffer=BytesIO()
                c = pycurl.Curl()
                c.setopt(pycurl.CAINFO, certifi.where()) # ssl 인증 건너뛰기
                c.setopt(c.URL, 'https://api.xforce.ibmcloud.com/ipr/history/'+ip)
                c.setopt(pycurl.HTTPHEADER,['Accept: application/json','Authorization: Basic ZmY0OWRkYzgtZDZkZS00ZjNlLWI0YjYtNjQ5NjE0N2Q1ZWZkOjQxNjZhNmJmLWE3M2MtNDU0My04MGQzLTYwNTFlMjAxMmU2NA=='])
        
                c.setopt(c.WRITEDATA, buffer)
                c.perform()
                c.close
            
                body = buffer.getvalue()
                blj = json.loads(body.decode('iso-8859-1'))            
                for i in blj['history'] :                 
                    if len(blj['history']) <= 1 :
                        tot = 0.0
                    else :                
                        tot = round(tot+blj['history'][1]['score'],1)
                ipr_li = [rej['country']['code'], tot]            
                #return ipr_li
                u_bl_ibm = 'update aw_ip_cache set cc=%s, bl_ibm=%s where ip = %s'
                curs.execute(u_bl_ibm,(ipr_li[0],ipr_li[1],ip))
                print(ip + "bl ibm 이 추가로 등록되었습니다")
                conn.commit()
                return        
        except :
            print('ipr_basic error',rej)        
            ipr_li =['ERR', 0]
            #return ipr_li
            u_bl_ibm = 'update aw_ip_cache set cc=%s, bl_ibm=%s where ip = %s'
            curs.execute(u_bl_ibm,(ipr_li[0],ipr_li[1],ip))
            conn.commit()
            return        


def bl():    
    s_ip = 'select ip from aw_ip_cache where bl_dg_yn = 1 and bl_ibm is null'
    curs.execute(s_ip) 
    rows = curs.fetchall()
    for ip in rows :
        ipr_basic_susp(ip[0])

def wl():
    day = datetime.date.today()
    s_ip = 'select ip from aw_ip_cache where cc=%s and bl_ibm = 0 and (web_ok >20 or fw_pd > 20) and (web_rej <20 and fw_db <5 and waf_ok <5 and waf_rej <5 and ips_pd <5 and ips_db <5) and times > 5'
    i_bl = 'insert into aw_bl (ip, flag, write_date) values (%s,%s,%s)'
    u_bl = 'update aw_ip_cache set bl_dg_yn = 3 where ip = %s'
    curs.execute(s_ip,('kr',)) 
    rows = curs.fetchall()
    for ip in rows :
        ipr_basic_susp(ip[0])
        curs.execute(i_bl,(ip[0],3,day))
        curs.execute(u_bl,(ip[0],))
    conn.commit()
    
        

wl()
conn.close()

