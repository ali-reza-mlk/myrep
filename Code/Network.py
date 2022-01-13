# -*- coding: utf-8 -*-
"""
Created on Thu May 27 01:02:45 2021

@author: alireza
"""
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint

#%%

def createModel():
    ch = 64
    T = 960
    inputShapeU = (T, ch,1)
    inputShapeX = (64,64,10)
    inputX = layers.Input(shape = inputShapeX)
    inputU = layers.Input(shape = inputShapeU)
    numClass = 4
    
    ############# U stream
    temporalU = layers.Conv2D(20, (10,1), padding='valid')(inputU)
    temporalU = layers.BatchNormalization()(temporalU)
    
    spatialU  = layers.Conv2D(15, (1,64), padding='valid')(temporalU)
    spatialU = layers.ELU(alpha=1.0)(spatialU)
    
    U1 = layers.MaxPooling2D(pool_size=(3,1), strides=None, padding="valid")(spatialU)
    permU1 = layers.Permute((1,3,2))(U1)
    
    U2 = layers.Conv2D(40, (10,15), padding='valid')(permU1)
    # U2 = layers.BatchNormalization()(U2)
    U2 = layers.ELU(alpha=1.0)(U2)
    permU2 = layers.Permute((1,3,2))(U2)
    
    U3 = layers.MaxPooling2D(pool_size=(3,1), strides=None, padding="valid")(permU2)
    U3 = layers.Conv2D(30, (12,40), padding='valid')(U3)
    U3 = layers.BatchNormalization()(U3)
    U3 = layers.ELU(alpha=1.0)(U3)
    permU3 = layers.Permute((1,3,2))(U3)
    
    U4 = layers.MaxPooling2D(pool_size=(3,1), strides=None, padding="valid")(permU3)
    U4 = layers.Conv2D(10, (15,30), padding='valid')(U4)
    U4 = layers.BatchNormalization()(U4)
    U4 = layers.ELU(alpha=1.0)(U4)
    permU4 = layers.Permute((1,3,2))(U4)
    
    U5 = layers.MaxPooling2D(pool_size=(3,1), strides=(2,1), padding="valid")(permU4)
    
    flatU = layers.Flatten()(U5)
    
    
    ############# X stream
    
    compX = layers.Conv2D(10, (2,2), padding='valid', strides=(2,2))(inputX)
    # compX = layers.Conv2D(4, (3,3), padding='valid')(inputX)
    # X1 = layers.MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(compX)
    X1 = layers.ELU(alpha=1.0)(compX)
    X1 = layers.Conv2D(3, (3,3), padding='valid', strides=(2,2))(X1)
    X1 = layers.BatchNormalization()(X1)
    X1 = layers.MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(X1)
    
    flatX = layers.Flatten()(X1)
    flatX = layers.Dropout(0.5)(flatX)
    ############# Dense
    
    XU = layers.Concatenate(axis=1)([flatX])
    denseIn = layers.Dropout(0.5)(XU)
    in1 = layers.Dense(20)(denseIn)
    outputs = layers.Dense(numClass, activation="softmax")(in1)
    
    model = keras.Model([inputU, inputX], outputs)
    model.summary()
    
    return model


#%% Data Generator

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=5, dimX=(138,10), dimU=(960,64),
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dimX = dimX
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.dimU = dimU
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        q = self.batch_size
        indexes = self.indexes[index*q:(index+1)*q]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        U, y = self.__data_generation(list_IDs_temp)
        
        return U, y #[X,U]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        # X = np.empty((self.batch_size*5, *self.dimX))
        U = np.empty((self.batch_size*5, *self.dimU,1))
        y = np.empty((self.batch_size*5), dtype=int)

        dest = 'D:\\Project\\BCI\\batches'
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            with open(dest+'\\batch'+str(ID)+'.pkl', 'rb') as f:
                loaded_obj = pickle.load(f)
            # Store sample
            for b in range(5):
                # X[b+5*i,] = loaded_obj['batchX'][b]
                U[b+5*i,:,:,:] = loaded_obj['batchU'][b] 
                # Store class
                y[b+5*i] = int(np.argmax(loaded_obj['batchL'][b]))
                
        U += np.random.randn(U.shape[0],U.shape[1],U.shape[2],U.shape[3])/50
        return U, keras.utils.to_categorical(y, num_classes=self.n_classes)


#%%


# weights_path = "weights"
# checkpoint = [ModelCheckpoint(filepath=weights_path, save_best_only=True)]

# model.compile(loss="categorical_crossentropy",
#               optimizer="adam", metrics=["accuracy"])

# trainSec = np.arange(1200)
# validSec = np.arange(1200,1831)
# training_generator = DataGenerator(trainSec)
# validation_generator = DataGenerator(validSec)

# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,epochs=200, 
#                     callbacks=checkpoint)

def fit_model(model, X, y, input_length=480, batch_size=32, epochs=30, callbacks=None):

    return model.fit(X, y, validation_split = .2,
        epochs=epochs, batch_size = 32, callbacks=callbacks
    )
