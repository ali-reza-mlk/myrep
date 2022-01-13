import gc
import nndata
import numpy as np
import pickle
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
#from keras.layers.recurrent import LSTM
from tensorflow.keras import regularizers
from tensorflow.keras import layers



def create_raw_model(nclasses, nchan, trial_length=480, l1=1e-5):
    """
    CNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((30, 1), strides=(15, 1)))
    model.add(Flatten())
    model.add(Dense(80, activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    gc.collect()
    return model

def create_raw_model2(nchan, nclasses, trial_length=960, l1=0, full_output=False):
    """
    CRNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((5, 1), strides=(5, 1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(40, activation="sigmoid", dropout=0.25, return_sequences=full_output))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model


# def create_raw_model3(nchan, nclasses, trial_length=960, l1=0, full_output=False):
#     """
#     CRNN model definition
#     """
#     inputShapeU = (trial_length, nchan,1)
#     # inputShapeX = (138,10,1)
#     # inputX = layers.Input(shape = inputShapeX)
#     inputU = layers.Input(shape = inputShapeU)
    
#     ############# U stream
#     temporalU = layers.Conv2D(20, (10,1), padding='valid')(inputU)
#     temporalU = layers.BatchNormalization()(temporalU)
    
#     spatialU  = layers.Conv2D(15, (1,64), padding='valid')(temporalU)
#     spatialU = layers.ELU(alpha=1.0)(spatialU)
    
#     U1 = layers.MaxPooling2D(pool_size=(3,1), strides=None, padding="valid")(spatialU)
#     permU1 = layers.Permute((1,3,2))(U1)
    
#     U2 = layers.Conv2D(40, (10,15), padding='valid')(permU1)
#     # U2 = layers.BatchNormalization()(U2)
#     U2 = layers.ELU(alpha=1.0)(U2)
#     permU2 = layers.Permute((1,3,2))(U2)
    
#     U3 = layers.MaxPooling2D(pool_size=(3,1), strides=None, padding="valid")(permU2)
#     U3 = layers.Conv2D(30, (12,40), padding='valid')(U3)
#     U3 = layers.BatchNormalization()(U3)
#     U3 = layers.ELU(alpha=1.0)(U3)
#     permU3 = layers.Permute((1,3,2))(U3)
    
#     U4 = layers.MaxPooling2D(pool_size=(3,1), strides=None, padding="valid")(permU3)
#     U4 = layers.Conv2D(10, (15,30), padding='valid')(U4)
#     U4 = layers.BatchNormalization()(U4)
#     U4 = layers.ELU(alpha=1.0)(U4)
#     permU4 = layers.Permute((1,3,2))(U4)
    
#     U5 = layers.MaxPooling2D(pool_size=(3,1), strides=(2,1), padding="valid")(permU4)
    
#     flatU = layers.Flatten()(U5)
#     denseIn = layers.Dropout(0.5)(flatU)
#     in1 = layers.Dense(20)(denseIn)
#     outputs = layers.Dense(nclasses, activation="softmax")(in1)
    
#     model = keras.Model(inputU, outputs)
#     model.summary()
#     model.compile(loss="categorical_crossentropy",
#               optimizer="adam", metrics=["accuracy"])
#     return model



def fit_model(model, X, y, train_idx, test_idx, input_length=50, batch_size=32, epochs=30, steps_per_epoch=1000, callbacks=None):    
    gc.collect()
    return model.fit_generator(
        nndata.crossval_gen(X,y, train_idx, input_length, batch_size),
        validation_data=nndata.crossval_test(X, y, test_idx, input_length),
        steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks                          
    )




# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, batch_size=5, dimX=(138,10), dimU=(960,64),
#                  n_classes=4, shuffle=True):
#         'Initialization'
#         self.dimX = dimX
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.dimU = dimU
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         q = self.batch_size
#         indexes = self.indexes[index*q:(index+1)*q]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         U, y = self.__data_generation(list_IDs_temp)
        
#         return U, y #[X,U]

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim)
#         # Initialization
#         # X = np.empty((self.batch_size*5, *self.dimX))
#         U = np.empty((self.batch_size*5, *self.dimU,1))
#         y = np.empty((self.batch_size*5), dtype=int)

#         dest = 'D:\\Project\\BCI\\batches'
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             with open(dest+'\\batch'+str(ID)+'.pkl', 'rb') as f:
#                 loaded_obj = pickle.load(f)
#             # Store sample
#             for b in range(5):
#                 # X[b+5*i,] = loaded_obj['batchX'][b]
#                 U[b+5*i,:,:,:] = loaded_obj['batchU'][b] 
#                 # Store class
#                 y[b+5*i] = int(np.argmax(loaded_obj['batchL'][b]))
                
#         U += np.random.randn(U.shape[0],U.shape[1],U.shape[2],U.shape[3])/50
#         return U, keras.utils.to_categorical(y, num_classes=self.n_classes)





