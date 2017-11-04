from slacker import Slacker

import websocket
from io import BytesIO
import pycurl
import os


websocket.enableTrace(True)

token = 'xoxp-223552584658-223479290179-225145163495-48232ac95a2792354c8382d3dc98a128'

slack = Slacker(token)
def slack_message(channel, message):    
    slack.chat.post_message(channel, message)

#curl -F file=@dramacat.gif -F channels=#general
'''
buffer = BytesIO()
c = pycurl.Curl()
c.setopt(c.file, '2017-08-10_perceptron_report.xls')
c.setopt(c.channels, '#general')
c.setopt(c.token, token)
c.setopt(c.URL,'https://slack.com/api/files.upload')

c.setopt(c.WRITEDATA, buffer)
c.perform()
c.close()
'''
#curl -F file=@dramacat.gif -F channels=general -F token=xoxp-223552584658-223479290179-225145163495-48232ac95a2792354c8382d3dc98a128 https://slack.com/api/files.upload'
os.system('curl -F file=@2017-08-03_perceptron_report.xls -F channels=general -F token=xoxp-223552584658-223479290179-225145163495-48232ac95a2792354c8382d3dc98a128 https://slack.com/api/files.upload')









#slack_message('#general','김극동')

#response = slack.rtm.start()
#print(response)
#sock_endpoint = response.body['url']

#slack_socket=websocket.create_connection(endpoint)

#slack_socket.recv()
