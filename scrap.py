import datetime
import threading
from bitmex_websocket import BitMEXWebsocket
import logging
import time
from time import sleep

art = """
 .d8888b.                        888                  d88888888888
d88P  Y88b                       888                 d88888  888  
888    888                       888                d88P888  888  
888       888d888888  88888888b. 888888 .d88b.     d88P 888  888  
888       888P"  888  888888 "88b888   d88""88b   d88P  888  888  
888    888888    888  888888  888888   888  888  d88P   888  888  
Y88b  d88P888    Y88b 888888 d88PY88b. Y88..88P d8888888888  888  
 "Y8888P" 888     "Y8888888888P"  "Y888 "Y88P" d88P     8888888888
                      888888                                      
                 Y8b d88P888                                      
                  "Y88P" 888                              by UmeW

             ***** Market Data Saver ******               
"""

print(art)

Coins = ["XBTUSD","XBTZ17","XBTH18","BCHZ17","DASHZ17","ETHZ17","ETC7D","LTCZ17","XMRZ17","ZECZ17"]

class NewTradeObs():

    F = datetime.timedelta(minutes=1)

    def __init__(self,symb):
        self.t = 0
        self.data = []
        self.symbol = symb
        self.lastTrade = time.time()

    def agregate(self):
        minPrice = self.data[0][1]
        maxPrice = self.data[0][1]
        buyCount = 0
        volume = 0
        av = 0
        for d in self.data:
            price = d[1]
            if price < minPrice:
                minPrice = price
            if price > maxPrice:
                maxPrice = price
            av += price
            volume += d[2]
            if d[3] == "Buy":
                buyCount += 1

        av = av / len(self.data)
        bsRatio = (buyCount * 100) / len(self.data)

        print(str(self.data[0][0]) + " : " + str(av) + ", low: " + str(minPrice) + ", high: " + str(maxPrice) + ", r: " + str(bsRatio) + ", vol:" + str(volume))
        with open('scrap/'+self.symbol+'.data','a') as f:
            f.write(str(self.data[0][0]) + " " + str(av) + " " + str(minPrice) + " " + str(maxPrice) + " " + str(int(bsRatio)) + " " + str(volume) + '\n')

        return self.data[0][0], av, minPrice, maxPrice, bsRatio, volume


    def newTrade(self, timestamp, price, size, orderType):
        x = timestamp + " " + str(price) + " " + orderType + " " + str(size)
        #print(x)
        #2017-12-08T00:13:29.332Z
        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        #print(dt.replace(second=0,microsecond=0))
        if self.t == 0 :
            self.curTime = dt.replace(second=0,microsecond=0)

        if dt > self.curTime + NewTradeObs.F :
            self.curTime = dt.replace(second=0,microsecond=0)
            self.agregate()
            self.data.clear()
            self.t = 0
        self.lastTrade = time.time()
        self.data.append([self.curTime, price, size, orderType])
        self.t += 1    

def run(symb):
    while True:
        try: 
            obs = NewTradeObs(symb)
            ws = BitMEXWebsocket(obs, endpoint="wss://www.bitmex.com/realtime", symbol=symb,
                                api_key=None, api_secret=None)
            while(ws.ws.sock.connected):
                sleep(5)
        except:
            pass        


for c in Coins:
    t = threading.Thread( target = run, args = (c, ) )
    t.start()