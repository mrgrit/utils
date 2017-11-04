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
        
    def orderByValue(self,key,value):
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
    
if __name__ == '__main__':
    ht = HashTable(100)
    
    print(ht)
    
    ht.add('bob',2)
    ht.add(11,3)
    ht.add('11',4)
    
    ht.add(' ',5)
    ht.add('test',[5,2,{'11',55}])
    print(ht)
    
    print()
    print(ht.lookUp(11))
    print(ht.lookUp(20))

    #date_ht = HashTable(100)
    #sip_ht = HashTable(5000)
    #dip_ht = HashTable(1000)
    #sport_ht = 
    
    #print(ht.printDistribution())
