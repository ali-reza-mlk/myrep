import numpy as np
import mne

#%% functions


"""
    not used in this case
"""

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


"""
    data initial normalization
"""
def sigDecomp(inputSig):
  inputSig = np.transpose(inputSig)
  mSig = np.mean(inputSig,0)
  vSig = np.var(inputSig,0)
  
  normSig = (inputSig - mSig) / np.sqrt(vSig)
  
  return normSig


"""
    select different time ranges from each trial to makes more 
"""
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


def dataPreprocess():

    path = '../data/files'
    
    
    method1 = [4,8,12]  # T1: Left fist , T2: both fists
    method2 = [6,10,14] # T1: Right fist , T2: both feet
    
    numSub = 109
    
    subTrainDataU = []
    label = []
    subIndex = []
    for sub in range(1,numSub+1):    ##### Number of subjects: 109
        
    
        for task in method1:     ##### Number of tasks: 14
        
            file = path + '/S'+str(sub).zfill(3)+'/S'+str(sub).zfill(3)+'R'+str(task).zfill(2)+'.edf'
            
            data = mne.io.read_raw_edf(file)   #### reads data
            
            raw_data = data.get_data()
            
            taskTime = data.annotations.duration
            taskLabel= data.annotations.description
            taskTime = np.append(0,taskTime)
            info = data.info
            channels = data.ch_names
            Fs = info['sfreq']
            
            raw_data = (raw_data - np.reshape(np.mean(raw_data,1),(64,1)))/ np.reshape(np.std(raw_data,1),(64,1))
            if task in method1:
                
                for i in range(len(taskTime)-1):
                  t1 = int(np.sum(taskTime[0:i+1]) * Fs)
                  t2 = int(np.sum(taskTime[0:i+2]) * Fs)
                  if (t2-t1)>Fs*3:
                    ranges = rangeT(t2-t1)
                    for ran in range(len(ranges)):
                        tmp = raw_data[:,ranges[ran][0]+t1:ranges[ran][1]+t1]
                        U = sigDecomp(tmp)
                        
                        
                        if taskLabel[i]=='T1' :
                            subTrainDataU.append(U)
                            label += [[1,0]]
                            subIndex +=[sub]
                        elif taskLabel[i]=='T2':
                            subTrainDataU.append(U)
                            label += [[0,1]]
                            subIndex += [sub]
            
            
            """
                for more classes (>2) - not used
            """
            if task in method2:
                
                for i in range(len(taskTime)-1):
                  t1 = int(np.sum(taskTime[0:i+1]) * Fs)
                  t2 = int(np.sum(taskTime[0:i+2]) * Fs)
                  if (t2-t1)>Fs*3:
                    ranges = rangeT(t2-t1)
                    for ran in range(len(ranges)):
                        tmp = raw_data[:,ranges[ran][0]+t1:ranges[ran][1]+t1]
                        U = sigDecomp(tmp)
                        
                        
                        if taskLabel[i]=='T1' :
                            subTrainDataU.append(U)
                            label += [[0,0,1]]
                            subIndex +=[sub]
                        elif taskLabel[i]=='T2':
                            subTrainDataU.append(U)
                            label += [[0,0,0,1]]
                            subIndex += [sub]
        
    subTrainDataU = np.array(subTrainDataU)
    label = np.array(label)
    subIndex = np.array(subIndex)
    
    return subTrainDataU, label, subIndex
    
