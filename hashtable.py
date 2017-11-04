import MySQLdb

class Entry:
    def __init__(self,key, value):
        self.key = key
        self.value = value

class HashTable:
    def __init__(self,numBuckets):
        if type(numBuckets) != int or numBuckets < 0:
            numBuckets = 1
        self.numBuckets = numBuckets
        self.buckets = [None for x in range(0,self.numBuckets)]
        
    def __str__(self):
        string = '{'
        for bucket in self.buckets:
            if bucket != None:
                for entry in bucket:
                    if type(entry.key) ==str:
                        string += ("'%s': "%entry.key)
                    else:
                        string += ("%s: "%entry.key)
                    if type(entry.value) ==str:
                        string += ("'%s', "%entry.value)
                    else:
                        string += ("%s, "%entry.value)
        return string.rstrip(', ') + "}"
    
    def hash(self,key):
        if not key:
            return None
        power = 0
        hashing = 0
        for char in key:
            hashing += (ord(char)-32)*pow(95,power)
            power += 1
        index = hashing % self.numBuckets
        return index
    
    def add(self,key,value):
        index = self.hash(str(key))
        if index < 0 or index > self.numBuckets:
            return False
        if self.buckets[index] == None:
            self.buckets[index] = [Entry(key,value)]
            return True
        else:
            for entry in self.buckets[index]:
                if entry.key == key:
                    entry.value = value
                    return True
            self.buckets[index].append(Entry(key,value))
            return True
    
    def updateValue(self,key,value):
        index = self.hash(str(key))
        if index == None:
            return False
        if self.buckets[index] == None:
            #print("Key Not Found!")
            return False
        else:
            for entry in self.buckets[index]:
                if entry.key == key:
                    entry.value = value
                    return True
            #print("Key Not Found!")
            return False
    
    def delete(self,key):
        index = self.hash(str(key))
        if index == None:
            return False
        if self.buckets[index] == None:
            #print("Key Not Found!")
            return False
        else:
            for entry in self.buckets[index]:
                if entry.key == key:
                    self.buckets[index].remove(entry)
                    return True
            #print("Key Not Found!")
            return False
    
    def lookUp(self,key):
        index = self.hash(str(key))
        if index == None:
            return False
        if self.buckets[index] == None:
            #print("Key Not Found!")
            return False
        else:
            for entry in self.buckets[index]:
                if entry.key == key:
                    return entry.value
            #print("Key Not Found!")
            return False
        
    def printDistribution(self):
        string = ''
        bucketNum = 0
        MIN = None
        MAX = 0
        TOTAL = 0
        for bucket in self.buckets:
            if bucket != None:
                #string += ("Bucket Number %d has: "%bucketNum)
                count = 0
                for entry in bucket:
                    count += 1
                #string += ("%d entries\n"%count)
                TOTAL += count
                if count > MAX:
                    MAX = count
                if MIN == None or count < MIN:
                    MIN = count
            else:
                pass
                #string += ("Bucket Number %d has 0 entries\n"%(bucketNum))
            bucketNum +=1
        string += ("Largest Bucket has %d entries\n"\
                       "Smallest Bucket has %d entries\nTotal entries: %d\n"\
                       "Avg bucket size is %f"%(MAX,MIN,TOTAL,(TOTAL/self.numBuckets)))
        return string

def check_time(f):
    def g(*a, **k):
        x = time.time()
        r = f(*a, **k)
        y = time.time()
        print(y-x)
        return r
    return g


#@check_time    
if __name__ == '__main__':
    conn = MySQLdb.connect('localhost','root','rhrnak#33','ais', charset = "utf8")
    curs = conn.cursor()
    ht = HashTable(1000)
    ht2 = HashTable(1000)
    ht3 = HashTable(1000)
    ht4 = HashTable(1000)
    ht5 = HashTable(1000)
    
    print(ht)
    
    ht.add('bob',[2,3,'aaa'])
    ht.add(11,3)
    ht.add('11',4)
    
    ht.add(' ',5)
    print(ht)
    
    print()
    print(ht.lookUp(11))
    #print(ht.printDistribution())

    sql = 'select ip, cc, ip_gubun, bl_ibm, bl_dg_yn, block_yn, web_ok, web_rej, fw_pd, fw_db, waf_ok, waf_rej, ips_pd, ips_db from aw_ip_cache'

    curs.execute(sql)
    rows = curs.fetchall()

    for i,row in enumerate(rows):
        try :
            ht.add(row[0],[row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]])
            if i <= 1000 or i >=2000:
                ht2.add(row[0],[row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]])
            elif i <= 2000 or i >=3000 :
                ht3.add(row[0],[row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]])
            elif i <= 3000 or i >=4000 :
                ht4.add(row[0],[row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]])
            elif i <= 4000 or i >=5000 :
                ht5.add(row[0],[row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]])
            else: break
        except:
            continue

    print(ht)
    #print(ht2)
    #print(ht3)
    #print(ht4)
    #print(ht5)

    ip = input()
    print(ht.lookUp(ip))
    





    
