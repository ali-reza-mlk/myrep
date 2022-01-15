import pickle
import numpy as np

# EEGNet-specific imports
from EEGmodels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from utils import dataPreprocess


K.set_image_data_format('channels_last')


subTrainDataU, label, subIndex = dataPreprocess()
l = subTrainDataU.shape[0]



index = np.random.permutation(l)
subTrainDataU = subTrainDataU[index]
label = label[index]
subIndex = np.array(subIndex)
subIndex = subIndex[index]

l = int(np.floor(l*0.75))


kernels, chans, samples = 1, 64, 480

# take 75/25 percent of the data to train/validate
X_train      = subTrainDataU[0:l,]
Y_train      = label[0:l]
X_validate   = subTrainDataU[l:,]
Y_validate   = label[l:]
S_validate   = subIndex[l:]
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

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='checkpoint_all.h5', verbose=1,
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
model.load_weights('checkpoint_all.h5')

numSub=109
accArray = np.zeros(numSub)
for sub in range(1,numSub+1):
    subData = np.where(S_validate==sub)
    validData = X_validate[subData[0]]
    pred = model.evaluate(validData, Y_validate[subData[0]], verbose=0)
    print(pred)
    accArray[sub-1] = pred[1]


print(accArray)

with open('accArray_all.pkl', 'wb') as f:
    pickle.dump(accArray, f)



