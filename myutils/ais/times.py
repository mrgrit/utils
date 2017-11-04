import datetime

a = datetime.datetime.strptime('11:46:39','%H:%M:%S')
b = datetime.datetime.strptime('14:38:12','%H:%M:%S')

r = b-a

print(r)



def addSecs(tm, secs):
    fulldate = datetime.date.timetuple(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(seconds=secs)
    print(fulldate.time())


a = datetime.datetime.now().time()
b = addSecs(a,r)
