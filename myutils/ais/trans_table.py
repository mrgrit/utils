import MySQLdb


conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")

def trans_ip_cache():
    curs = conn.cursor()
    #sel_sql = 'select ip_cache_no, ip, cc, ip_gubun, times, bl_ibm, write_time, modi_time, write_date,seq, web_ok, web_rej,fw_pd,fw_db,waf_ok,waf_rej,ips_pd,ips_db from aw_ip_cache'
    #ins_sql = 'insert into ip_cache set ip_cache_no=%s, ip=%s, cc=%s, ip_gubun=%s, times=%s, bl_ibm=%s, write_time=%s, modi_time=%s, write_date=%s,seq=%s, web_ok=%s, web_rej=%s,fw_pd=%s,fw_db=%s,waf_ok=%s,waf_rej=%s,ips_pd=%s,ips_db=%s'
    sel_sql = 'select ip, cc, ip_gubun, times, bl_ibm, write_time, modi_time, write_date,seq, web_ok, web_rej,fw_pd,fw_db,waf_ok,waf_rej,ips_pd,ips_db from aw_ip_cache'
    ins_sql = 'insert into ip_cache set ip=%s, cc=%s, ip_gubun=%s, times=%s, bl_ibm=%s, write_time=%s, modi_time=%s, write_date=%s,seq=%s, web_ok=%s, web_rej=%s,fw_pd=%s,fw_db=%s,waf_ok=%s,waf_rej=%s,ips_pd=%s,ips_db=%s'
    curs.execute(sel_sql)
    rows = curs.fetchall()
    for r in rows:
        #curs.execute(ins_sql,(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15],r[16],r[17]))
        curs.execute(ins_sql,(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15],r[16]))

    conn.commit()

def trans_log_full():
    curs = conn.cursor()
    sel_sql = 'select log_full_no, log_type, log_time, source_ip, dest_ip, dest_port, result, method, get_post, attack_code, times, agent, write_time, modi_time, alarm_yn, ip_cache_no, write_date from log_full'
    ins_sql = 'insert into aw_log_full set log_full_no=%s, log_type=%s, log_time=%s, source_ip=%s, dest_ip=%s, dest_port=%s, result=%s, method=%s, get_post=%s, attack_code=%s, times=%s, agent=%s, write_time=%s, modi_time=%s, alarm_yn=%s, ip_cache_no=%s, write_date=%s'
    curs.execute(sel_sql)
    rows = curs.fetchall()
    for r in rows:
        curs.execute(ins_sql,(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15],r[16]))

    conn.commit()

trans_ip_cache()
#trans_log_full()
conn.close()
                     

