from datetime import date
from datetime import datetime


def write_log(msg) :    
    #log_file = logdir+str(logday)+logfile
    now = datetime.now()
    day = date.today()
    f = open('./'+str(day)+'_log.txt','a')
    log_message = '['+ str(now) + ']  ' + msg + '\n'
    f.write(log_message)
    f.close()

    
def get_logday() :
    logday_list = []
    logday = ''
    today = str(date.today())
    today = today.split('-')        
    logday = str(today[0]) + str(today[1])+str(today[2])        
    return logday
