import pickle
import numpy as np

from utils import dataPreprocess

# EEGNet-specific imports
from EEGmodels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold



K.set_image_data_format('channels_last')

numSub = 109
kernels, chans, samples = 1, 64, 480

subTrainDataU, label, subIndex = dataPreprocess()

l = subTrainDataU.shape[0] #### number of trials

subPermutation = np.random.permutation(numSub)+1

accArray = np.zeros(numSub)

kf = KFold(n_splits=4)


for train_index, test_index in kf.split(subPermutation):
    
    trialTrainIndex = []
    trialTestIndex  = []
    for i in np.random.permutation(l):
        if subIndex[i] in subPermutation[test_index]:
            trialTestIndex += [i]
        else:
            trialTrainIndex += [i]


    X_train      = subTrainDataU[trialTrainIndex,]
    Y_train      = label[trialTrainIndex]
    X_validate   = subTrainDataU[trialTestIndex,]
    Y_validate   = label[trialTestIndex]
    S_validate   = subIndex[trialTestIndex]
############################# EEGNet portion ##################################


    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples,
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 8, F2 = 16, 
                   dropoutType = 'Dropout')


    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    # count number of parameters in the model
    numParams    = model.count_params()
    print("number of params: ", numParams)
    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='checkpoint_part.h5', verbose=1,
                                   save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
    class_weights = {0:1, 1:1}

################################################################################
    
    fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 100,
                            verbose = 2, validation_data=(X_validate, Y_validate),
                            callbacks=[checkpointer], class_weight = class_weights)
    
    # load optimal weights
    model.load_weights('checkpoint_part.h5')
    
    numSub=109
    
    for sub in subPermutation[test_index]:
        subData = np.where(S_validate==sub)
        validData = X_validate[subData[0]]
        pred = model.evaluate(validData, Y_validate[subData[0]], verbose=0)
        print(pred)
        accArray[sub-1] = pred[1]
    
    
print(accArray)

with open('accArray_partial.pkl', 'wb') as f:
    pickle.dump(accArray, f)



