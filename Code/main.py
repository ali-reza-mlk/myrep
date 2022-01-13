# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:09:34 2021

@author: alireza
"""
import os
import numpy as np
import mne
import pickle

#%% functions


def chebFilter(x):
  N , ch = x.shape
  W = np.matmul(np.transpose(x),x) / N
  wSum = np.sum(W,1)
  
  D = np.diag(wSum)
  L = D - W
  
  w, v = np.linalg.eig(L)
  wMax = max(abs(w))
  w = 2 * w / wMax - 1
  w = np.diag(w)
  X = np.zeros((len(L),len(L),5))
  
  for i in range(5):
    c = np.zeros(len(L+1))
    c[i+1]=1
    tmp = np.polynomial.chebyshev.chebval(w, c)
    tmp = np.matmul(v,tmp)
    X[:,:,i] = np.matmul(tmp,np.transpose(v))

  return X

def sigDecomp(inputSig):
  inputSig = np.transpose(inputSig)
  mSig = np.mean(inputSig,0)
  vSig = np.var(inputSig,0)
  
  normSig = (inputSig - mSig) / np.sqrt(vSig)
  
  return normSig

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
    if m==1:
        out = np.empty((len(labels),64,64,5))
    elif m==0:
        out = np.empty((len(labels),3*fs,64))
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
        
        Tr += [[1*fs,4*fs]]
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

#%% Upload data


path = '../data/files'


method1 = [4,8,12]  # T1: Left fist , T2: both fists
method2 = [6,10,14] # T1: Right fist , T2: both feet

numSub = 109


for sub in range(1,numSub+1):    ##### Number of subjects: 109
    
    subTrainrainDataX = []
    subTrainrainDataU = []
    label = []
    for task in range(1 ,15):     ##### Number of tasks: 14
    
        file = path + '/S'+str(sub).zfill(3)+'/S'+str(sub).zfill(3)+'R'+str(task).zfill(2)+'.edf'
        
        data = mne.io.read_raw_edf(file)   #### reads data
        
        raw_data = data.get_data()
        
        taskTime = data.annotations.duration
        taskLabel= data.annotations.description
        taskTime = np.append(0,taskTime)
        info = data.info
        channels = data.ch_names
        Fs = info['sfreq']
        #%
        # mne.viz.plot_raw(data,n_channels=5,scalings=.0001) # Ploting a sample of data
        
        # data.plot_psd(fmin=2., fmax=80., average=True, spatial_colors=False) # Freq resp
        
    #% Filter data
        fil_data = mne.filter.notch_filter(raw_data, Fs, 60,filter_length=220
                                           ,trans_bandwidth=5);  # Notch filter 60 Hz
        
        fil_data = mne.filter.filter_data(fil_data, Fs, 2, None, picks=None, filter_length='auto',
                               l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                               method='fir', iir_params=None, copy=True,phase='zero',
                               fir_window='hamming', fir_design='firwin',pad='reflect_limited');
        


        if task in method1:
            
            for i in range(len(taskTime)-1):
              t1 = int(np.sum(taskTime[0:i+1]) * Fs)
              t2 = int(np.sum(taskTime[0:i+2]) * Fs)
              if (t2-t1)>Fs*3:
                ranges = rangeT(t2-t1)
                for ran in range(len(ranges)):
                    tmp = fil_data[:,ranges[ran][0]+t1:ranges[ran][1]+t1]
                    U = sigDecomp(tmp)
                    
                    
                    if taskLabel[i]=='T1' :
                        subTrainrainDataU.append(U)
                        label += [0]
                    elif taskLabel[i]=='T2':
                        subTrainrainDataU.append(U)
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
                    U = sigDecomp(tmp)
                    
                    
                    if taskLabel[i]=='T1' :
                        subTrainrainDataU.append(U)
                        label += [2]
                    elif taskLabel[i]=='T2':
                        subTrainrainDataU.append(U)
                        label += [3]
                    else:
                        label += [4]
                    
    
    with open('../proData/train-sub'+str(sub)+'.pkl', 'wb') as f:
        pickle.dump({'trainU':subTrainrainDataU,'trainL':label}, f)
        
#%%

def batchCreate(path, batchSize):
    dest = '../batches'
    numSub = 109
    perm = np.random.permutation(np.arange(1,1+numSub))
    fs=160
    resU = np.empty((0,3*fs,64))
    resL = np.empty(0)
    batchNum = 0
    for i in range(int(numSub/3)):
        tmpU = resU
        tmpL = resL
        if i<int(numSub/3)-1:
            k=3
        else:
            k=4
        for sub in perm[i*3:i*3+k]:
            with open(path+'/train-sub'+str(sub)+'.pkl', 'rb') as f:
                loaded_obj = pickle.load(f)
            f.close()
            os.remove(path+'/train-sub'+str(sub)+'.pkl')
            tmpU = np.concatenate((tmpU,loaded_obj['trainU']))
            tmpL = np.concatenate((tmpL,loaded_obj['trainL']))
        
        permj = np.random.permutation(len(tmpL))
        L = int(len(tmpL)/batchSize)
        for j in range(L):
            batchU = [tmpU[permj[x]] for x in range(j*batchSize,(1+j)*batchSize)]
            batchL = [tmpL[permj[x]] for x in range(j*batchSize,(1+j)*batchSize)]
            
            with open(dest+'/batch'+str(batchNum)+'.pkl', 'wb') as f:
                pickle.dump({'batchU':batchU,'batchL':batchL}, f)
            
            batchNum += 1
        if L*batchSize<len(tmpL):
            resU = [tmpU[permj[x]] for x in range(L*batchSize,len(tmpL))]
            resL = [tmpL[permj[x]] for x in range(L*batchSize,len(tmpL))]
        else:
            resU = np.empty((0,3*fs,64))
            resL = np.empty(0)
            
        print('batch '+str(batchNum))
    
path='../proData'
batchSize = 5
batchCreate(path, batchSize)


