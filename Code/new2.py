# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:09:08 2021

@author: alireza
"""


import os
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from sklearn import linear_model
#%% functions



def dataSep(labels):
    
    L = len(labels)

    trainLabel = []
    perm = np.random.permutation(L)
    count=0
    for i in range(L):
        
        if label[perm[i]]!=4:
            trainLabel += [perm[i]]
        elif label[perm[i]]==4 and count<round(L/8):
            trainLabel += [perm[i]]
            count+=1

            
    trainLabel = [int(i) for i in trainLabel]
    return trainLabel 

def arrnData(inX, labels, m):
    fs = 160
    if m==0:
        out = np.empty((len(labels),64,63))
    else:
        out = np.empty(len(labels))
    
    for i in range(len(labels)):
        out[i,] = inX[labels[i]]
    return out



def rangeT(L):
    
    fs = 160
    Tr =[[0,3*fs]]
    if L<=4*fs:
        Tr += [[L-3*fs,L]]
        
    elif L>4*fs and L<= 5*fs:
        
        Tr += [[fs,4*fs]]
        Tr += [[L-3*fs,L]]
        
    elif L>5*fs and L<=6*fs:
        
        Tr += [[1*fs,4*fs]]
        Tr += [[2*fs,5*fs]]
        Tr += [[L-3*fs,L]]
        
    else:
        
        Tr += [[1*fs,4*fs]]
        Tr += [[2*fs,5*fs]]
        Tr += [[3*fs,6*fs]]
        Tr += [[L-3*fs,L]]
        
    return Tr

def pinky(x):
    
    a = mne.time_frequency.psd_array_welch(x,Fs, fmin=1, fmax=40, n_fft=258,
                                n_overlap=0, n_per_seg=None, n_jobs=1, 
                                average='mean', window='hamming')
    
    aa = np.log(a[0])
    f  = np.reshape(np.log(a[1]), (len(a[1]),1))
    for i in range(64):
        reg = linear_model.LinearRegression()
        
        b = aa[i]
        reg.fit(f,b)
        r = reg.predict(f)
        aa = aa - r
    return aa


#%% Upload data

os.chdir('D:\Project\BCI\Dataset')

path = 'D:\\Project\\BCI\\Dataset\\files'


method1 = [3,4,7,8,11,12]  # T1: Left fist , T2: both fists
method2 = [5,6,9,10,13,14] # T1: Right fist , T2: both feet

numSub = 1

Udata=np.empty((675*3,64,63))
Ldata=np.empty(675*3)
for sub in range(1,numSub+3):    ##### Number of subjects: 109
    
    subTrainrainDataU = []
    label = []
    for task in range(1 ,15):     ##### Number of tasks: 14
    
        file = path + '\\S'+str(sub).zfill(3)+'\\S'+str(sub).zfill(3)+'R'+str(task).zfill(2)+'.edf'
        
        data = mne.io.read_raw_edf(file)   #### reads data
        
        raw_data = data.get_data()
        
        taskTime = data.annotations.duration
        taskTime = np.append(0,taskTime)
        info = data.info
        channels = data.ch_names
        Fs = info['sfreq']
        
        # mne.viz.plot_raw(data,n_channels=5,scalings=.0001) # Ploting a sample of data
        
        # data.plot_psd(fmin=2., fmax=80., average=True, spatial_colors=False) # Freq resp

        fil_data = raw_data
    
        
    #%% Epoch

        if task in method1:
            
            for i in range(len(taskTime)-1):
              t1 = int(np.sum(taskTime[0:i+1]) * Fs)
              t2 = int(np.sum(taskTime[0:i+2]) * Fs)
              if (t2-t1)>Fs*3:
                ranges = rangeT(t2-t1)
                for ran in range(len(ranges)):
                    tmp = fil_data[:,ranges[ran][0]+t1:ranges[ran][1]+t1]
                    U = pinky(tmp)
                    
                    subTrainrainDataU.append(U)
                    if ((i+1)%2)==0 and ((i+1)%4)>0 :
                        label += [0]
                    elif not ((i+1)%4):
                        label += [1]
                    else:
                        label += [4]
        
        if task in method2:
            
            for i in range(len(taskTime)-1):
              t1 = int(np.sum(taskTime[0:i+1]) * Fs)
              t2 = int(np.sum(taskTime[0:i+2]) * Fs)
              if (t2-t1)>Fs*3:
                ranges = rangeT(t2-t1)
                for ran in range(len(ranges)):
                    tmp = fil_data[:,ranges[ran][0]+t1:ranges[ran][1]+t1]
                    U = pinky(tmp)
                    subTrainrainDataU.append(U)
                    if ((i+1)%2)==0 and ((i+1)%4)>0 :
                        label += [2]
                    elif not ((i+1)%4):
                        label += [3]
                    else:
                        label += [4]
                    
    totlabel = dataSep(label)
    
    trainU = arrnData(subTrainrainDataU,totlabel,0)
    tLabel = arrnData(label,totlabel,2)
    Udata[675*(sub-1):675*sub] = trainU
    Ldata[675*(sub-1):675*sub] = tLabel

#%%

for ch in [6]:
    sel = [[],[],[],[],[]]
    for i in range(56):
        
        sel[int(tLabel[i])]+= [trainU[i,ch]]
    
    m = [[],[],[],[],[]]
    v = [[],[],[],[],[]]
    fig , ax = plt.subplots(nrows=2, ncols=2)
    for i in range(5):
        sel[i] = np.array(sel[i])
    
        m[i] = np.mean(sel[i],0)
        v[i] = np.std(sel[i],0)
        
    k=-1
    for i in range(5):
        if i != 4:
            k+=1
            ind1 =  k % 2
            ind0 = int(k/2)
            ax[ind1][ind0].errorbar(np.linspace(1, 40,len(m[4])) , m[4], yerr=v[4])
            ax[ind1][ind0].errorbar(np.linspace(1, 40,len(m[i])) , m[i], yerr=v[i])
            ax[ind1][ind0].set_title(str(i))
