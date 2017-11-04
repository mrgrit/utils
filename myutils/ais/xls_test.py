import pandas as pd
from pandas.io import sql
#import numpy as np
import MySQLdb
#from sqlalchemy import create_engine

xls_file = pd.ExcelFile('fw_ac_full.xls')

df = xls_file.parse('fw_ac_full')

df = df.iloc[1:,[1,4,5,7,8,11,12]]

#print(str(df.iloc[2:3,:]))

conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")

sql.write_frame(df, con=conn, name = 'fw_full', if_exists='append', flavor='mysql')







#for i,df_data in enumerate(df) :
#    print(i, df_data[i])

    #insertSQL = 'insert into fw_full (log_time, source_ip, source_port, destination_ip, destination_port, protocol, result,write_date) values (?,?,?,?,?,?,?,?)'

    #curs.execute(select_sql,(day,'fw'))


