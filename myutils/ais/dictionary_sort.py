import operator

ibm_top = []
fw_top = []
waf_ips = []
dic_data = []
dic_tmp1 = dict()
dic_tmp2 = dict()
dic_tmp3 = dict()


dic_tmp1 = {'ip':'111','cc':'kr','ibm_score':10,'times':20,'day':30}
dic_tmp2 = {'ip':'222','cc':'us','ibm_score':20,'times':30,'day':10}
dic_tmp3 = {'ip':'333','cc':'cn','ibm_score':30,'times':10,'day':20}

dic_data.append(dic_tmp1)
dic_data.append(dic_tmp2)
dic_data.append(dic_tmp3)

print(dic_data)


day_top = sorted(dic_data, key=lambda k: k['day'], reverse = True)

#ibm_top = sorted(dic_data, key=operator.itemgetter('ibm_score'))

for i in range(2) :
    print(day_top[i])
message = str(day_top[i]['ip']+ ' : 국가 : '+ day_top[i]['cc']+ 'IBM Score : '+
              str(day_top[i]['ibm_score'])+ '접속시도 횟수 : '+str(day_top[i]['times'])+
              '지속 일수 : '+ str(day_top[i]['day']))

print(message)

