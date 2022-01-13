import numpy as np
import io
import mne
import gc
import pickle

def load_edf_signals(Path):
    try:
        sig = mne.io.read_raw_edf(Path)
        sigbuf = sig.get_data()
        
        # (n,3) annotations: [t in s, duration, type T0/T1/T2]
        annotations = sig.annotations
    except KeyboardInterrupt:
        # prevent memory leak and access problems of unclosed buffers
        raise
        
    del sig
    return sigbuf.transpose(), annotations

def load_physionet_data(subject_id, num_classes=2, long_edge=False):
    """
    subject_id: ID (1-109) for the subject to be loaded from file
    num_classes: number of classes (2, 3 or 4) for L/R, L/R/0, L/R/0/F
    long_edge: if False include 1s before and after MI, if True include 3s

    returns (X, y, pos, fs)
        X: Trials with shape (N_subjects, N_trials, N_samples, N_channels)
        y: labels with shape (N_subjects, N_trials, N_classes)
        pos: 2D projected electrode positions
        fs: sample rate
    """
    SAMPLE_RATE = 160
    EEG_CHANNELS = 64
    
    BASELINE_RUN = 2
    MI_RUNS = [4, 8, 12] # l/r fist
    if num_classes >= 4:
        MI_RUNS += [6, 10, 14] # feet (& fists)
        
    # total number of samples per long run
    RUN_LENGTH = 125 * SAMPLE_RATE 
    # length of single trial in seconds
    TRIAL_LENGTH = 3 if not long_edge else 10
    NUM_TRIALS = 21 * num_classes 
    
    n_runs = len(MI_RUNS)
    gc.collect()
    X = np.zeros((n_runs, RUN_LENGTH, EEG_CHANNELS))
    events = []

    base_path = '../data/files/S%03d/S%03dR%02d.edf'
    
    for i_run, current_run in enumerate(MI_RUNS):
        # load from file
        Path = base_path % (subject_id, subject_id, current_run)
        signals, annotations = load_edf_signals(Path)  
        X[i_run,:signals.shape[0],:] = signals
        
        # read annotations
        current_event = [i_run, 0, 0, 0] # run, class (l/r), start, end
        
        for annotation in annotations:
            t = int(annotation['duration'] * SAMPLE_RATE)
            action = int(annotation['description'][1])
            
            if action == 0 and current_event[1] != 0:
                # make 6 second runs by extending snippet
                length = TRIAL_LENGTH * SAMPLE_RATE
                pad = (length - (t - current_event[2])) / 2
                current_event[2] -= int(pad + (t-current_event[2]) % 2)
                current_event[3] = int(t + pad)
                if (current_run - 6) % 4 != 0 or current_event[1]==2:
                    if (current_run - 6) % 4 == 0:
                        current_event[1] = 3
                    events.append(current_event)
            elif action > 0:
                current_event = [i_run, action, t, 0]
    
    # split runs into trials 
    num_mi_trials = len(events)
    trials = np.zeros((NUM_TRIALS, TRIAL_LENGTH * SAMPLE_RATE, EEG_CHANNELS))
    labels = np.zeros((NUM_TRIALS, num_classes))
    
    for i,ev in enumerate(events):
        tmp = X[ev[0], ev[2]:ev[3],:]
        trials[i, :, :] = (tmp - np.mean(tmp,0)) / np.std(tmp,0)
        labels[i, ev[1] - 1] = 1.
    
    if num_classes < 3:
        with open('../proData/train-sub'+str(subject_id)+'.pkl', 'wb') as f:
            pickle.dump({'U':trials,'L':labels}, f)
            
        return (trials[:num_mi_trials,...], labels[:num_mi_trials,...], 
                projection_2d(get_physionet_electrode_positions()), SAMPLE_RATE)
    else:
        # baseline run
        Path = base_path % (subject_id, subject_id, BASELINE_RUN)
        signals, annotations = load_edf_signals(Path)    
        SAMPLES = TRIAL_LENGTH * SAMPLE_RATE
        for i in range(num_mi_trials, NUM_TRIALS):
            offset = np.random.randint(0, signals.shape[0]-SAMPLES)
            tmp = signals[offset:offset+SAMPLES, :]
            trials[i, :, :] = (tmp - np.mean(tmp,0)) / np.std(tmp,0)
            labels[i, -1] = 1.
        
        with open('../proData/train-sub'+str(subject_id)+'.pkl', 'wb') as f:
           pickle.dump({'U':trials,'L':labels}, f)

        if (trials==None).any() or (trials==0).any():
            ArithmeticError



    return trials, labels, projection_2d(get_physionet_electrode_positions()), SAMPLE_RATE

PHYSIONET_ELECTRODES = { 
    1 : "FC5", 2 : "FC3", 3 : "FC1", 4 : "FCz", 5 : "FC2", 6 : "FC4", 
    7 : "FC6", 8 : "C5", 9 : "C3", 10: "C1", 11: "Cz", 12: "C2", 
    13: "C4", 14: "C6", 15: "CP5", 16: "CP3", 17: "CP1", 18: "CPz", 
    19: "CP2", 20: "CP4", 21: "CP6", 22: "Fp1", 23: "Fpz", 24: "Fp2",
    25: "AF7", 26: "AF3", 27: "AFz", 28: "AF4", 29: "AF8", 30: "F7", 
    31: "F5", 32: "F3", 33: "F1", 34: "Fz", 35: "F2", 36: "F4", 
    37: "F6", 38: "F8", 39: "FT7", 40: "FT8", 41: "T7", 42: "T8", 
    43: "T9", 44: "T10", 45: "TP7", 46: "TP8", 47: "P7", 48: "P5", 
    49: "P3", 50: "P1", 51: "Pz", 52: "P2", 53: "P4", 54: "P6", 
    55: "P8", 56: "PO7", 57: "PO3", 58: "POz", 59: "PO4", 60: "PO8",
    61: "O1", 62: "Oz", 63: "O2", 64: "Iz"
}

def get_physionet_electrode_positions():

    refpos = get_electrode_positions()
    return np.array([refpos[PHYSIONET_ELECTRODES[idx]] for idx in range(1,65)])

def projection_2d(loc):
    """
    Azimuthal equidistant projection (AEP) of 3D carthesian coordinates. 
    Preserves distance to origin while projecting to 2D carthesian space.

    loc: N x 3 array of 3D points
    returns: N x 2 array of projected 2D points
    """
    x, y, z = loc[:,0], loc[:,1], loc[:,2]
    theta = np.arctan2(y, x) # theta = azimuth
    rho = np.pi / 2 - np.arctan2(z, np.hypot(x,y)) # rho = pi/2 - elevation

    return np.stack((
        np.multiply(rho, np.cos(theta)), 
        np.multiply(rho, np.sin(theta))
    ), 1)

def get_electrode_positions():
    """
    Returns a dictionary (Name) -> (x,y,z) of electrode name in the extended
    10-20 system and its carthesian coordinates in unit sphere.
    """
    positions = dict()
    with io.open("electrode_positions.txt", "r") as pos_file:
        for line in pos_file:
            parts = line.split()
            positions[parts[0]] = tuple([float(part) for part in parts[1:]])
    return positions


def batchCreator(Path, dest, mode):
    
    X = np.zeros((109*84,960,64,1))
    y = np.zeros((109*84,4))
    
    for i in range(1,110):
        with open(Path+'\\train-sub'+str(i)+'.pkl', 'rb') as f:
            loaded_obj = pickle.load(f)
        f.close()
        for j in range(84):
            X[(i-1)*84+j,:,:,0] = loaded_obj['U'][j,:,:]
            
        y[(i-1)*84:i*84] = loaded_obj['L']
        
        del loaded_obj
        gc.collect()
        
    if mode:
        index = np.random.permutation(len(y))

    else:
        
        index = np.random.permutation(109)
        tmp   = np.empty(len(y))
        for i in range(109):
            tmp[i*84:(i+1)*84] = np.arange(index[i]*84,(1+index[i])*84)
            indexExt = [int(tmp[i]) for i in range(len(tmp))]
        
        print(index)
        
    
    for i in range( int(len(y)/6)):
            
        batchU = X[indexExt[i*6:(i+1)*6]]
        for j in range(5):
            for k in range(64):
                batchU[j,:,k,0] -= np.mean(batchU[j,:,k,0])
                batchU[j,:,k,0] /= np.std(batchU[j,:,k,0])
                
        batchL = y[indexExt[i*6:(i+1)*6]]
        if mode:
            with open(dest+'\\batch'+str(i)+'.pkl', 'wb') as f:
                pickle.dump({'batchU':batchU,'batchL':batchL}, f)
        else:
            with open(dest+'\\batch'+str(i)+'.pkl', 'wb') as f:
                pickle.dump({'batchU':batchU,'batchL':batchL, 'sub': index[int(i/14)]}, f)
        
        gc.collect()
    


def chebFilter(x):
  N , ch = x.shape
  W = np.matmul(np.transpose(x),x) / N
  wSum = np.sum(W,1)
  # tucker_rank =  [10, 10, 6]
  D = np.diag(wSum)
  L = D - W
  
  w, v = np.linalg.eig(L)
  wMax = max(abs(w))
  w = 2 * w / wMax - 1
  w = np.diag(w)
  Xx = np.zeros((64,64,10))
  
  for i in range(10):
    c = np.zeros(len(L+1))
    c[i+1]=1
    tmp = np.polynomial.chebyshev.chebval(w, c)
    tmp = np.matmul(v,tmp)
    Xx[:,:,i] = np.matmul(tmp,np.transpose(v))
    Xx[:,:,i] = Xx[:,:,i] - np.diag(np.diag(Xx[:,:,i]))
    
  # _, A = parafac(X, rank=10, init='random', tol=10e-6)
  # C = np.concatenate((A[0]/np.max(abs(A[0])),A[1]/np.max(abs(A[1])),A[2]/np.max(abs(A[2]))),0)
    
  return Xx



def sigDecomp(inputSig):

  nn = inputSig.shape
  print(nn)
  Ux = np.zeros((nn[0],64,64,10))
  for l in range(nn[0]):
      mSig = np.mean(inputSig[l],0)
      vSig = np.std(inputSig[l],0)
      
      # cenSig = inputSig - mSig
      normSig = (inputSig[l] - mSig) / vSig
      
      # u, s, vh = np.linalg.svd(cenSig, full_matrices=False)
      
      # U = np.matmul(u , vh)
      
      if vSig.any()==0:
          print(1)
          continue
      Xc = chebFilter(normSig)
      Ux[l] = Xc
      
  
  
  return Ux
