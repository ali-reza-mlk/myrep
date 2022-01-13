
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:28:55 2021

@author: Alireza
"""

import os

import gc
#from tensorflow import keras
import models1
import nndata
import models
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import Network
SPLITS = 4
numCh = 64
input_length = 3 * 160 # = 3s
electrodes = range(64)
epochs = 5
epoch_steps = 5 # record performance 5 times per epoch
batch = 20
numSub = 109
#nclasses = [2, 3, 4]
Nclasses = [4]
splits = range(4)
# splits = [0]


#%%
for j,nclasses in enumerate(Nclasses):
    try:
        del X,y
    except:
        pass
    gc.collect()
    X,y = nndata.load_raw_data(electrodes=electrodes, num_classes=nclasses)


    Xv = np.zeros((X.shape[0]* X.shape[1], X.shape[2], X.shape[3],X.shape[4]))
    yv = np.zeros((X.shape[0]* X.shape[1],4))

    for i in range(X.shape[0]):
       Xv[i*X.shape[1]:(i+1)*X.shape[1]] = X[i]
       yv[i*X.shape[1]:(i+1)*X.shape[1]] = y[i]


    ranIndex = np.random.permutation(Xv.shape[0])
    Xv = Xv[ranIndex]
    yv = yv[ranIndex]

    numC = np.sum(yv,0)
    Xt = np.zeros((int(numC[0]+numC[2]), Xv.shape[1], Xv.shape[2], Xv.shape[3]))
    yt = np.zeros((int(numC[0]+numC[2]),2))
    k = 0
    for i in range(yv.shape[0]):
       if yv[i,0]==1 or yv[i,1]==1:
           Xt[k] = Xv[i]
           yt[k] = yv[i,[0,1]]
           k += 1
           if np.std(Xv[i])<.9:
             print(np.std(Xv[i]))

    del Xv, yv

    model = models.create_raw_model(2, numCh, trial_length=480, l1=0)


    l = Xt.shape[0]
    lt = int(np.floor(l*0.7))
    model.fit(Xt[:lt], yt[:lt], validation_data = (Xt[lt:],yt[lt:]), epochs=10, batch_size=32)
