import MySQLdb
import datetime
import xlrd

conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")

def bl_batch() :
    day = datetime.date.today()
    curs = conn.cursor()
    file_location = './bl.xlsx'
    workbook = xlrd.open_workbook(file_location)
    sheet = workbook.sheet_by_index(0)
    sql = "insert into aw_bl (ip, write_date,flag) values(%s,%s,%s)"

    for r in range(sheet.nrows):        
        ip=sheet.cell_value(r,0)
        print(ip)
        try:
            curs.execute(sql,(ip,day,1))
        except :
            print(ip + "[ERROR]")
            continue
    conn.commit()
    return 0


bl_batch()
conn.close()

