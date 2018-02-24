import numpy as np
np.random.seed(7)
import tensorflow as tf
import datetime, time, threading, math, random
random.seed(7)
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import *
from keras.layers import *
from keras import backend as K
from enum import Enum
from time import sleep

l_input = Input(batch_shape=(None,3))
l_dense = Dense(2, activation='tanh')(l_input)
out = Dense(3, activation='softmax')(l_dense)
model = Model(inputs=[l_input], outputs=[out])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
X=np.array([])
X.shape = (0,3)
Y=np.array([])
Y.shape = (0,3)
for b in range(10):
    b = b/10
    for p in range(100):
        p = p/100
        for c in range(1):
            c = p * c
            X = np.vstack([X, [b,p,c] ])
            y = [0,0,0]
            if (b > p and p < 0.5):
                y[0] = 1
            elif(c > 0 and p > 0.5 ):
                y[1] = 1
            else:
                y[2] = 1
            Y = np.vstack([Y,y])

model.fit(X, Y, epochs=60, batch_size=30)


scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

