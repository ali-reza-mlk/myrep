
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:28:55 2021

@author: Alireza
"""

import os
# from importlib import reload
import gc
#from tensorflow import keras
import models1
import nndata
import models
# import util
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import util
import pickle

SPLITS = 4
numCh = 64
input_length = 3 * 160 # = 3s
electrodes = range(64)
epochs = 5
epoch_steps = 5 # record performance 5 times per epoch
batch = 20
numSub = 109
#nclasses = [2, 3, 4]
nclasses = [4]
splits = range(4)
# splits = [0]


#%%
results = np.zeros((len(nclasses), len(splits), 4, epochs*epoch_steps))
for j,nclasses in enumerate(nclasses):
    try:
        del X,y
    except:
        pass
    gc.collect()
    X,y = nndata.load_raw_data(electrodes=electrodes, num_classes=nclasses)
    
    print(X.shape) 
    # del X
    
    # for sid in range(109):
    #     with open('D:\\Project\\BCI\\proData\\train-sub'+str(sid+1)+'.pkl', 'rb') as f:
    #         loaded_obj = pickle.load(f)
    #         f.close()
        
    #     with open('D:\\Project\\BCI\\proData\\train-U-sub-'+str(sid+1)+'.pkl', 'wb') as f:
    #         pickle.dump({'Ux': util.sigDecomp(loaded_obj['U'][sid,:,:,:,0])}, f)
    #         f.close()



    

    steps_per_epoch = np.prod(X.shape[:2]) / batch * (1-1./SPLITS) / epoch_steps
    for ii,i in enumerate(splits):


        dis = '%d CLASS, SPLIT %d' % (nclasses, i)
        print(dis)
        idx = list(range(len(X)))
        train_idx, test_idx = nndata.split_idx(i, 5, idx)

        model = models.create_raw_model(
            nchan=len(electrodes),
            nclasses=4,
            trial_length=input_length
        )
        model.summary
        #save best weights for each model
        weights_path = "weights-%dcl-%d.hdf5" % (nclasses,i)
        checkpoint = [ModelCheckpoint(filepath=weights_path, save_best_only=True)]

        # run training
        h = models.fit_model(
            model, X, y, train_idx, test_idx, input_length=input_length, 
            batch_size=batch,  steps_per_epoch=steps_per_epoch, epochs=epochs*epoch_steps, 
            callbacks=checkpoint
        )
"""
        # save training history
        results[j, ii, :, :] = [
            h.history["loss"], 
            h.history["val_acc"], 
            h.history["val_loss"] 
        ]
"""        

a = y[0]
print(sum(a))

#%% Subject retraining


# # Using pretrained weights, this procedure loops over all subjects and refines
# # the parameters for each subject recording the new accuracy.

# input_length = 480
# electrodes = range(64)
# classes = 4

# # X, y = nndata.load_raw_data(electrodes=electrodes, num_classes=classes)

# weights_file = "weights-4cl-%d.hdf5"
# model = models.create_raw_model(
#     nchan=len(electrodes), 
#     nclasses=classes, 
#     trial_length=input_length
# )

# results = np.zeros((len(X), 3))

# for split in range(5):
#     idx = np.arange(len(X))
#     train_idx, test_idx = nndata.split_idx(split, SPLITS, idx)
#     Xsub, ysub = nndata.crossval_test(X, y, test_idx, input_length, flatten=False)
    
#     # for each subject in the training set
#     for i, subject_X in enumerate(Xsub):
#         current_subject = test_idx[i]
#         subject_y = ysub[i,:]
#         print(current_subject)
        
#         # accuracy without retraining
#         model.load_weights(weights_file % (split))
#         tmp = model.evaluate(
#             subject_X.reshape((-1,) + (input_length, len(electrodes), 1)), 
#             subject_y.reshape((-1, classes)), verbose=0)
        
#         results[current_subject, 0] = tmp[1]
#         # retrain and validate on 4 subject splits
#         epochs = 5
#         temp_results = [] 
#         for subject_split in range(4):
#             trial_idx = np.arange(len(subject_X))
#             train_trials, test_trials = nndata.split_idx(subject_split, 4, trial_idx)

#             model.load_weights(weights_file % (split))
#             h = model.fit(
#                 subject_X[train_trials,:], subject_y[train_trials,:], 
#                 validation_data=(subject_X[test_trials,:], subject_y[test_trials,:]),
#                 epochs=10, batch_size=2, verbose=0
#             )
#             # save best validation accuracy
#             temp_results.append(np.max(h.history["val_acc"]))
#         results[current_subject, 1] = np.mean(temp_results)


#%%

# import matplotlib.pyplot as plt

# plt.plot(results[:,0], label= 'Random subject selection')
# plt.plot(results[:,1], label= 'Retrained network bu subjects')
# plt.legend()



#%% Shuffling data and creating batches



# Now delete the variables

"""
classes = 4
import models1

Path = 'D:/Project/BCI/proData'

dest = 'D:/Project/BCI/batches'
# util.batchCreator(Path, dest, mode=0)

# weights_path = "weights-mixSub/weights-4cl-%d.hdf5"
# checkpoint = [ModelCheckpoint(filepath=weights_path % (split), save_best_only=True)]

model = models1.create_raw_model3(
            nchan=len(electrodes), 
            nclasses=nclasses, 
            trial_length=input_length*2)

trainSec = np.arange(1050)
validSec = np.arange(1050,1526)
training_generator = models1.DataGenerator(trainSec)
validation_generator = models1.DataGenerator(validSec)

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,epochs=5)
#%%
model.save_weights('weights.h5') 

#%%

# import pickle

# result1 = np.zeros(109)
# for i in range(1050,1526):
#         with open(dest+'\\batch'+str(i)+'.pkl', 'rb') as f:
#             loaded_obj = pickle.load(f)
#         f.close()
        
#         order = loaded_obj['sub']
#         tmp = model.predict(loaded_obj['batchU'])
        
#         a = np.argmax(loaded_obj['batchL'],1)
#         b = np.argmax(tmp,1)
#         result1[order] += sum(a == b)
        
# #%%
# # import matplotlib.pyplot as plt

# # plt.plot(result, label= 'Random subject selection')
# # plt.scatter(range(109),result1)
# # plt.legend()

# #%%
# resultAdd = np.zeros(109)
# resultAdd1 = np.zeros(109)
# import pickle
# for i in range(109):
    
#     model = models1.create_raw_model3(
#             nchan=len(electrodes), 
#             nclasses=nclasses, 
#             trial_length=input_length*2)
#     model.load_weights('weights.h5')
    
#     dest = 'D:/Project/BCI/batches'
    
    
#     X = np.empty((84,960,64,1))
#     y = np.empty((84,4))
#     for j in range(14):
#         with open(dest+'\\batch'+str(j+i*14)+'.pkl', 'rb') as f:
#             loaded_obj = pickle.load(f)
#             f.close()
#         X[j*6:(j+1)*6] = loaded_obj['batchU']
#         y[j*6:(j+1)*6] = loaded_obj['batchL']
        
#     order = loaded_obj['sub']
#     index= np.random.permutation(84)
    
#     xTrain, yTrain = X[index[:65]], y[index[:65]]
#     xTest , yTest  = X[index[65:]], y[index[65:]]
    
#     h = model.fit(
#                 xTrain, yTrain, 
#                 epochs=40, verbose=0
#             )

    # h = model.fit_generator(
    #     xTrain, np.argmax(yTrain),
    #     validation_data=(xTest, np.argmax(yTest)),
    #     epochs=10                         
    # )

    # resultAdd[i] = np.max(h.history["val_accuracy"])
    
    # print(np.max(h.history["accuracy"]))
    
    
    # z = model.evaluate(xTest,yTest)
    # resultAdd[order] = z[1]
    # if i>75:
    #     resultAdd1[order] = z[1]

#%%

# import matplotlib.pyplot as plt

# plt.plot(resultAdd, label= 'Random subject selection')
# plt.scatter(range(109),resultAdd1)
# plt.plot(result/84, label= 'Random subject selection')
# plt.scatter(range(109),result1/84)
# # plt.legend()


model = models.create_raw_model(
            nchan=len(electrodes), 
            nclasses=4, 
            trial_length=input_length
        )   
"""
