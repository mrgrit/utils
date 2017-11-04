import MySQLdb
import os

conn = MySQLdb.connect('localhost','root','rhrnak#33','graph', charset = "utf8")
curs = conn.cursor()
i = 0
country = dict()
with open("./cc.txt") as f:
    while True:
        line = f.readline()
        if not line : break
        
        else :
            if line =="\n" : continue
            print(line.strip("\n"), i)            
            if i==0 :
                country['name_kr']=line.strip("\n")
            elif i==1:
                country['name_eng']=line.strip("\n")
            elif i==2:
                country['cc']=line.strip("\n")
            i = i+1
            if i == 5 :                
                insert = 'insert into cc (name_eng, name_kr, cc) values(%s,%s,%s)'
                #curs.execute(insert,(str(country['name_eng']),str(country['name_kr']),str(country['cc'])))
                curs.execute(insert,(country['name_eng'],country['name_kr'],country['cc'])) 
                conn.commit()
                i =0
                country = {}
        
