import MySQLdb
import pandas as pd
import numpy as np
from datetime import date
import datetime


conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")

bl = pd.read_csv('bl.csv',names=['ip'])
#bl = np.array(bl.tolist())
print(bl)
curs = conn.cursor()
u_bl = 'update aw_ip_cache set block_yn = 1 where block_yn=0 and ip = %s'
for i,ip in enumerate(bl['ip']) :
    print(ip)
    curs.execute(u_bl,(ip, ))
    print('completed => ' ,i)
    conn.commit()

conn.close()
    
    
