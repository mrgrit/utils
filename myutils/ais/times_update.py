import MySQLdb

conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")


s_ip_sql = 'select ip from aw_ip_cache where write_date like %s'
s_count_sql = 'select count(*) from aw_log_full where source_ip = %s'
u_times_sql = 'update aw_ip_cache set times = count where ip = %s'

curs = conn.cursor()

curs.execute(s_ip_sql,('2017-03-2%',))
rows = curs.fetchall()
for ip in rows :
    curs.execute(s_count_sql,(ip[0],))
    times = curs.fetchone()
    u_times_sql = 'update aw_ip_cache set times = %s where ip = %s'
    curs.execute(u_times_sql,(times[0], ip[0]))
    print(ip[0], times[0])

conn.commit()
    


