import numpy as np
np.random.seed(7)
import tensorflow as tf
import datetime
import time
import threading
import math
import random
random.seed(7)
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import *
from keras.layers import *
from keras import backend as K
from enum import Enum
from time import sleep

THREADS = 8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(linewidth = 500) 

art = """
 .d8888b.                        888                  d88888888888
d88P  Y88b                       888                 d88888  888
888    888                       888                d88P888  888
888       888d888888  88888888b. 888888 .d88b.     d88P 888  888
888       888P"  888  888888 "88b888   d88""88b   d88P  888  888
888    888888    888  888888  888888   888  888  d88P   888  888
Y88b  d88P888    Y88b 888888 d88PY88b. Y88..88P d8888888888  888
 "Y8888P" 888     "Y8888888888P"  "Y888 "Y88P" d88P     8888888888
                      888888                  d88P
                 Y8b d88P888                 d88P
                  "Y88P" 888                d88P          by UmeW

           ****** Deep AC3 Trader ******
"""

# HyperParams
LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = 0.1 	# entropy coefficient
LEARNING_RATE = 1e-4
EPS_START = 0.5
EPS_END = 0.1
EPS_SLOPE = 600

N_STEP_RETURN = 8
MIN_BATCH = 32
NUM_HISTORY = 300
NUM_STATE = 1 * NUM_HISTORY + 1 + 1 + 1 + 1# Scrapped data + (Shares bought?) + (Budget?)
NUM_DENSE = 120
NUM_DENSE2 = 30
GAMMA = 0.99
GAMMA_N = GAMMA ** N_STEP_RETURN

CAN_SHORT = False
NUM_ACTIONS = 3  # Buy = 0 , Sell = 1 , Hold = 2

# States Var
mdPrice = []
mdPriceMin = []
mdPriceMax = []
mdBSRatio = []
mdVolume = []
mdVar = [0] * THREADS
mdMean =  [0] * THREADS
mdTimeMax =  [0] * THREADS
class Action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

aHistory = [[] for i in range(THREADS)]

stopSignal = False
testFile = open("result2.test", "a")
def loadData():
    j = 0
    for j in range(0, 8):
        with open('training2/training_'+ str(j) +'.data', 'r') as f:
            buf = f.readlines()
            mdPrice.append([])
            mdPriceMin.append([])
            mdPriceMax.append([])
            mdBSRatio.append([])
            mdVolume.append([])
            esp = 0
            esp2 = 0
            for line in buf:  # we should test if everything good at import
                dat = line.split(' ')
                #>>> t = "2017-12-08 23:22:00 16066.530120481928 16060 16072 38 225691"
                #['2017-12-08', '23:22:00', '16066.530120481928', '16060', '16072', '38', '225691']
                mdPrice[j].append(float(dat[2]))
                esp += float(dat[2])
                esp2 += float(dat[2]) ** 2
                mdPriceMin[j].append(float(dat[3]))
                mdPriceMax[j].append(float(dat[4]))
                mdBSRatio[j].append(float(dat[5]))
                mdVolume[j].append(float(dat[6]))
            
                
            mdTimeMax[j] = int(len(buf))
            esp = esp / mdTimeMax[j]
            esp2 = esp2 / mdTimeMax[j]

            mdVar[j] = math.sqrt(esp2 - (esp ** 2))
            mdMean[j] = esp
            #print(mdVar[j])


class Brain():

    def __init__(self):
        g = tf.Graph()
        SESSION = tf.Session(graph=g)
        self.session = SESSION
        with g.as_default():
            tf.set_random_seed(7)
            K.set_session(self.session)
            K.manual_variable_initialization(True)
            self.model = self.BuildModel()
            self.graph = self.BuildGraph()
            self.session.run(tf.global_variables_initializer())
            self.default_graph = tf.get_default_graph()
        #self.default_graph.finalize()

        self.buffer = [[], [], [], [], []]
        self.lock = threading.Lock()

    def BuildModel(self):
        l_input = Input(batch_shape=(None, NUM_STATE))
        #l_predense = Dense(NUM_DENSE, activation='relu', kernel_regularizer=regularizers.l2(0.01))(l_input)
        #l_dense = Dense(NUM_DENSE, activation='relu', kernel_regularizer=regularizers.l2(0.01))(l_predense)
        
        l_predense = Dense(NUM_DENSE, activation='tanh')(l_input)
        l_dense = Dense(NUM_DENSE, activation='tanh')(l_predense)
        
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()

        self.intermediateModel = Model(inputs=[l_input], outputs=[l_dense])
        self.intermediateModel._make_predict_function()
        return model

    def BuildGraph(self):
        s_t = tf.placeholder(tf.float64, shape=(None, NUM_STATE))
        r_t = tf.placeholder(tf.float64, shape=(None, 1))  # r + gamma vs'
        a_t = tf.placeholder(tf.float64, shape=(None, NUM_ACTIONS))
        p_t, v_t = self.model(s_t)

        advantage = r_t - v_t
        log_prob = tf.log(tf.reduce_sum(p_t * a_t, axis=1, keep_dims=True) + 1e-10)

        loss_policy = - log_prob * tf.stop_gradient(advantage)
        loss_value = LOSS_V * tf.square(advantage)

        entropy = LOSS_ENTROPY * tf.reduce_sum(p_t * tf.log(p_t + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        #loss_total = tf.reduce_mean(entropy)
        self.loss = loss_total
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def getPrediction(self, s):
        with self.default_graph.as_default():
            #print(self.intermediateModel.predict(s))
            p, v = self.model.predict(s)
            #print(p)
            #s_t, a_t, r_t, minimize = self.graph
            #k = self.session.run(self.entropy, feed_dict={s_t: s})
            #print(k) 
            return p, v

    def getValue(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def getPolicy(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def pushTraining(self, action, reward, oldStep, newStep, threadId):
        with self.lock:
            act = np.zeros(NUM_ACTIONS)
            act[action] = 1
            self.buffer[0].append(act)
            self.buffer[1].append(reward)
            self.buffer[2].append(oldStep)
            if newStep is None:
                 self.buffer[3].append(np.zeros(NUM_STATE))
                 self.buffer[4].append(0)
            else:     
                self.buffer[3].append(newStep)
                self.buffer[4].append(1)

    def optimize(self):
        if len(self.buffer[0]) > MIN_BATCH :
            batch = []
            with self.lock:
                batch = self.buffer
                self.buffer =  [[], [], [], [], []]
                #print(self.threadC)

            s_t, a_t, r_t, minimize = self.graph

            a = np.vstack(batch[0])
            r = np.vstack(batch[1])
            s = np.vstack(batch[2])
            newStates = np.vstack(batch[3])
            newStatesMask = np.vstack(batch[4])
            
            newStatesValue = self.getValue(newStates)

            rew = r + newStatesValue * GAMMA_N * newStatesMask 

            #x = np.hstack([s,a,r,newStatesValue,newStatesMask,rew])
            #print(x)
            #print(len(s))
            #if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
            #print("*************************************")
            #print(s)
            self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: rew})
            #for i in range(0,100):
            #    self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: rew})
            #    k = self.session.run(self.loss, feed_dict={s_t: s, a_t: a, r_t: rew})
            #    print(k)

class Optimizer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not stopSignal:
            sleep(0.001)
            brain.optimize()
            
class Actor(threading.Thread):

    def __init__(self, idt, isTest):
        threading.Thread.__init__(self)
        self.id = idt
        self.isTest = isTest
        self.steps = 0
        self.simCount = 0
        #print("Actor " + str(idt) +" created")

    def run(self):
        #print("Actor " + str(self.id) +" started")
        if self.isTest :
            self.startSimulation()
        else:
            while not stopSignal:
                sleep(0)
                self.startSimulation()
                self.simCount += 1
        #print(str(self.id) + " : " + str(self.steps))

    def normalizeBin(self, value):
        #if value > 0:
        #    return [1]
        #else:
        #    return [-1]

        if value > 255 : 
            return [0.9]*8

        r = value
        ret = [0.1] * 8
        i = 0
        while r > 0:
            x = (r % 2)*(0.8) + 0.1
            ret[i] = x
            r = int(r/2)
            i +=1
        return ret


    def startSimulation(self):
        if self.isTest:
            self.budget = 300
        elif self.simCount < 100:
            self.budget = random.randint(1000, 5000) *( self.id + 1)
        elif self.simCount < 200:
            self.budget = random.randint(9 *(100 - self.simCount) + 1000, 45 * (100 - self.simCount) + 5000) *( self.id + 1)
        else:
            self.budget = 300

        self.budget = random.randint(50000, 100000) *( self.id + 1)

        self.initbud = self.budget
        self.shares = 0
        self.mem = [] # (a, t, st, r)
        self.r = 0
        t = NUM_HISTORY - 1
        self.timeMax = mdTimeMax[self.id % 8] #random.randint(NUM_HISTORY , mdTimeMax[self.id % 8])
        if self.id == 12:
            t = self.simCount % ( mdTimeMax[self.id % 8] - NUM_HISTORY - 50) +  NUM_HISTORY - 1
            self.timeMax = t + 50
        totalSteps = self.timeMax - t
        self.R = 0
        #print("t " + str(t) +" timemax " + str(mdTimeMax))
        actions = []
        self.badActions = 0
        kill = False
        while t < self.timeMax - 1 and not kill:

            
            #st = ([[self.budget] + [self.shares] + mdPrice[t + 1 - NUM_HISTORY: t + 1] + mdPriceMin[t + 1 - NUM_HISTORY: t + 1] +
            #mdPriceMax[t + 1 - NUM_HISTORY: t + 1] + mdBSRatio[t + 1 - NUM_HISTORY: t + 1] + mdVolume[t + 1 - NUM_HISTORY: t + 1]])
            
            if self.budget >= mdPrice[self.id % 8][t] :
                canBuy =  [1]
            else :
                canBuy =  [-1]

            priceA = mdPrice[self.id % 8 ][t + 1 - NUM_HISTORY: t + 1]
            priceA = [(x - mdMean[self.id % 8 ]) / mdVar[self.id % 8 ] for x in priceA]
            st = [[(self.shares * mdPrice[self.id % 8 ][t])/mdVar[self.id % 8] ] + canBuy + [(self.budget )/ mdVar[self.id % 8], mdMean[self.id % 8 ]/mdVar[self.id % 8] ] +  priceA]
            state = np.array(st)
            #if self.isTest : print(state)

            a, v = self.getActionValue(state, t)
            if self.id == 12:
                #if self.isTest: print("self.budget: " + str(self.budget) + " - p: " + str(mdPrice[self.id % 8 ][t]))
                if self.budget > mdPrice[self.id % 8 ][t] and  mdPrice[self.id % 8 ][t] < mdMean[self.id % 8 ]:
                    a = 0
                elif self.shares > 0 and mdPrice[self.id % 8 ][t] > mdMean[self.id % 8 ]:
                    a = 1
                else :
                    a = 2 
            if self.isTest and False:
                print("time: " + str(t) + " | price: " + str(mdPrice[self.id][t]))
                print("budget: " + str(self.budget) +  "| shares:" + str(self.shares))
                print("action: " + Action(a).name)
                testFile.write(str(mdPrice[self.id][t])+ " "+ str(a)+ "\n")

            r = self.act(state, t, a)
            self.R += r
           # print("reward: " + str(r))
            self.pushTraining(a,t,st,r)
            #if(t%3 == 0):
            while (len(brain.buffer[0]) > MIN_BATCH and not stopSignal) :
                sleep(0)
            t += 1
            self.steps += 1
            actions.append(a)
            if(self.budget <= 0 and self.shares == 0):
                kill = True

        ratioComplete = (t*100)/(self.timeMax - 1)
        aHistory[self.id].append(actions)
        badActionPct = self.badActions
        print("Actor " + str(self.id) +" FINISH -- REWARD: " + str(self.R)+ " -- Bad: " + str(badActionPct) + " -- SimC: " + str(self.simCount)+ " -- Comp:%" + str(ratioComplete) )

    def pushTraining(self,a,t,st,r):
        # a debug
        self.mem.append((a,t,st,r))

        self.r = (self.r + GAMMA_N * r) / GAMMA
        #print("selfR: "  + str(self.r))
        if t == self.timeMax - 2:
            while len(self.mem) > 0: 
                brain.pushTraining(self.mem[0][0], self.r, self.mem[0][2], None, self.id)
                self.r = (self.r - self.mem[0][3])/GAMMA
                self.mem.pop(0)

        elif len(self.mem) == N_STEP_RETURN : 
            brain.pushTraining(self.mem[0][0], self.r, self.mem[0][2], st, self.id)
            self.r = self.r - self.mem[0][3]
            self.mem.pop(0)

        #print("NselfR: "  + str(self.r))
        #print("\n\n")

    def getActionValue(self, state, time):
        eps = self.getEps(self.steps)
        a, v = brain.getPrediction(state)
        #print("TIME: " + str(time) + " - EPS: " + str(eps) + " - VAL: " + str(v))
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1), v
        else:
            return np.argmax(a), v

    def act(self, state, time, action):
        
        action = Action(action)

        oldPortfolio = self.budget + self.shares * mdPrice[self.id % 8][time]
        oldBud = self.budget
        #print("oldportfolioValue: " + str(oldPortfolio))
        if action == Action.BUY and self.budget >= mdPrice[self.id % 8][time]:
            self.shares += 1
            self.budget -= mdPrice[self.id % 8][time]
        elif action == Action.BUY and self.budget < mdPrice[self.id % 8][time]:
            self.budget  -= mdPrice[self.id % 8][time] * 3
            self.badActions += 1    
        elif action == Action.SELL and (self.shares > 0 or CAN_SHORT):
            self.budget += mdPrice[self.id % 8][time] * self.shares
            self.shares = 0
        elif action == Action.SELL and (self.shares <= 0 and not CAN_SHORT):
            self.budget -= mdPrice[self.id % 8 ][time] * 3
            self.badActions += 1
        #elif action == Action.HOLD and self.shares == 0:
        #    self.budget -= mdPrice[self.id % 8][time] * 10
        #    self.badActions += 1


        newPortfolio = self.budget + self.shares * mdPrice[self.id % 8][time + 1]
        #print("newportfolioValue: " + str(newPortfolio))
        #return newPortfolio - oldPortfolio
        #return (self.budget - oldBud)/(mdVar[self.id % 8])
        return newPortfolio - oldPortfolio

    def getEps(self, time):
        if self.isTest : 
            return 0.0
        else:
            EPS_STEP = EPS_SLOPE *(self.timeMax - NUM_HISTORY) 
            #print("EPS STEP : " + str(EPS_STEP))
            if time >= EPS_STEP:
                return EPS_END
            else:
                slope = (EPS_END - EPS_START) / (EPS_STEP)
                eps = slope * time + EPS_START
                return eps


print(art)
loadData()




def start():
    global brain 
    global stopSignal
    stopSignal = False
    brain = Brain()                         
    # Start Actors
    actors = [Actor(i,False) for i in range(THREADS)]
    for t in actors: t.start()
    # Start Critics
    opt = Optimizer()
    opt.start()
    sleep(3600)
    stopSignal = True
    sleep(5)
    #mdTimeMax = 2 * mdTimeMax
    #Test Strategy
    #print("**TRAINING COMPLETE*********")
    testers = [Actor(i,True) for i in range(8)]
    results = [0] * 8
    for t in testers: t.start()
    for t in testers:
        t.join()
        results[t.id] = str(int(t.R))

    return " ".join(results)

if False : 
    HP_LOSS_V = [0.5]
    HP_LOSS_ENTROPY = [0.01,0.001,0.1,0.5,1,10]
    HP_LEARNING_RATE = [1e-4,5e-4,1e-3,1e-2]
    HP_EPS_START = [0.5,0.6,0.7,0.4,0.3]
    HP_EPS_END = [0.15,0.05,0.25,0.1]
    HP_EPS_SLOPE = [10, 15, 5]
    HP_N_STEP_RETURN = [8]
    HP_MIN_BATCH = [32,64,512]
    HP_NUM_HISTORY = [1,2,3]
    HP_GAMMA = [0.99]

    for loss_v in HP_LOSS_V:
        LOSS_V = loss_v
        for eps_start in HP_EPS_START:
            EPS_START = eps_start
            for eps_end in HP_EPS_END:
                EPS_END = eps_end
                for eps_slope in HP_EPS_SLOPE:
                    EPS_SLOPE = eps_slope
                    for n_step_return in HP_N_STEP_RETURN:
                        N_STEP_RETURN = n_step_return
                        for min_batch in HP_MIN_BATCH:
                            MIN_BATCH = min_batch
                            for num_history in HP_NUM_HISTORY:
                                NUM_STATE = 1 * NUM_HISTORY + 1 + 1 + 1 
                                HP_NUM_DENSE = [30, 10, 100]
                                for num_dense in HP_NUM_DENSE:
                                    NUM_DENSE = num_dense
                                    for loss_entropy in HP_LOSS_ENTROPY:
                                        LOSS_ENTROPY = loss_entropy
                                        for learning_rate in HP_LEARNING_RATE:
                                            LEARNING_RATE = learning_rate
                                            for gamma in HP_GAMMA:
                                                GAMMA = gamma
                                                GAMMA_N = GAMMA ** N_STEP_RETURN
                                                result = start()
                                                strin = ("loss_v: " + str(loss_v) + 
                                                " | loss_entropy: " + str(loss_entropy) +
                                                " | learning_rate: " + str(learning_rate) +
                                                " | eps_start: " + str(eps_start) +
                                                " | eps_end: " + str(eps_end) +
                                                " | eps_slope: " + str(eps_slope) +
                                                " | n_step_return: " + str(n_step_return) +
                                                " | min_batch: " + str(min_batch) +
                                                " | num_history: " + str(num_history) +
                                                " | num_dense: " + str(num_dense) + 
                                                " | result: " + str(result)  + "\n"
                                                )
                                                print(strin)

else :
    print(start())



testFile.close

plt.ion()
fig = plt.figure()
lines = []
prices = []
for i in range(THREADS):

    x = np.arange(NUM_HISTORY - 1, mdTimeMax[i%8], 1)
    priceA = mdPrice[i % 8 ][NUM_HISTORY - 1:  mdTimeMax[i % 8] ]
    priceA = np.array([(x - mdMean[i % 8 ])/ mdVar[i % 8 ] for x in priceA])
    prices.append(priceA)
    ax = fig.add_subplot(int(THREADS/2), 2, i + 1)

    acts = aHistory[i][0]
    fill = [-1]*(mdTimeMax[i] - NUM_HISTORY- len(acts) + 1)
    actions =  np.array( acts + fill ) + 1

    beee,line = ax.plot(x, priceA, 'b-', x, actions, 'ro')
    
    lines.append(line)
    plt.title(str(i))


fig.canvas.draw()


k = 0
while k < 1000:
    for i in range(THREADS):
        if k < len(aHistory[i]):
            acts =  aHistory[i][k]
            fill = [-1]*(mdTimeMax[i] - NUM_HISTORY - len(acts) + 1)
            actions =  np.array( acts + fill ) + 1
        else:
            acts = aHistory[i][-1]
            fill = [-1]*(mdTimeMax[i] - NUM_HISTORY - len(acts) + 1)
            actions =  np.array( acts  + fill ) + 1
        lines[i].set_ydata(actions)    
    k += 1
    t = time.time()
    while time.time() < t + 1 :        
        fig.canvas.flush_events()
        sleep(0.001)

#for phase in np.linspace(0, 10*np.pi, 500):
#    line1.set_ydata(np.sin(x + phase))
   
#    t = 0
#    while t < 1000:
#        fig.canvas.flush_events()
#        sleep(0.001)
#        t+=1

