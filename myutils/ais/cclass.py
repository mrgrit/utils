c_ = []
c_class = []

ip = "218.157.224.15"

c_ = ip.split('.')
c_class = c_[0] +'.'+ c_[1] +'.'+ c_[2] + '.0/24'

print(str(c_class))


