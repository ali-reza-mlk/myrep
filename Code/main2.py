import models
import nndata
import numpy as np
import pickle


from tensorflow.keras.callbacks import ModelCheckpoint

SPLITS = 5
input_length = 3 * 160 # = 3s
electrodes = range(64)
epochs = 3
epoch_steps = 5 # record performance 5 times per epoch
batch = 16
#nclasses = [2, 3, 4]
nclasses = [2]
splits = range(5)
#splits = [0]

results = np.zeros((len(nclasses), len(splits), 4, epochs*epoch_steps))
for j,nclasses in enumerate(nclasses):
    try:
        del X,y
    except:
        pass
    X,y = nndata.load_raw_data(electrodes=electrodes, num_classes=nclasses)
    
    steps_per_epoch = np.prod(X.shape[:2]) / batch * (1-1./SPLITS) / epoch_steps
    for ii,i in enumerate(splits):
        print("%d CLASS, SPLIT %d" % (nclasses, i))
        idx = np.arange(len(X))
        train_idx, test_idx = nndata.split_idx(i, 5, idx)

        model = models.create_raw_model(
            nchan=len(electrodes), 
            nclasses=nclasses, 
            trial_length=input_length
        )        
        
        # save best weights for each model
        weights_path = "weights-%dcl-%d.hdf5" % (nclasses,i)
        checkpoint = [ModelCheckpoint(filepath=weights_path, save_best_only=True)]
        
        # run training
        h = models.fit_model(
            model, X, y, train_idx, test_idx, input_length=input_length, 
            batch_size=batch,  steps_per_epoch=steps_per_epoch, epochs=epochs*epoch_steps, 
            callbacks=checkpoint
        )

        # save training history
        results[j, ii, :, :] = [ 
            h.history["acc"], 
            h.history["loss"], 
            h.history["val_acc"], 
            h.history["val_loss"] 
        ]
        
with open('result1.pkl', 'wb') as f:
   pickle.dump(results, f)
        

print("****************************************************************************************************")
        
SPLITS = 5

input_length = 3*160
electrodes = range(64)
classes = 2

X, y = nndata.load_raw_data(electrodes=electrodes, num_classes=classes)

weights_file = "weights-2cl-%d.hdf5"
model = models.create_raw_model(
    nchan=len(electrodes), 
    nclasses=classes, 
    trial_length=input_length
)

results = np.zeros((len(X), 2))

for split in range(5):
    idx = np.arange(len(X))
    train_idx, test_idx = nndata.split_idx(split, SPLITS, idx)
    Xsub, ysub = nndata.crossval_test(X, y, test_idx, seg_length=input_length, flatten=False)
    
    # for each subject in the training set
    for i, subject_X in enumerate(Xsub):
        current_subject = test_idx[i]
        subject_y = ysub[i,:]
        print(current_subject)
        
        # accuracy without retraining
        model.load_weights(weights_file % (split))
        results[current_subject, 0] = model.evaluate(
            subject_X.reshape((-1,) + (input_length, len(electrodes), 1)), 
            subject_y.reshape((-1, classes)), verbose=0
        )[1]
        
        # retrain and validate on 4 subject splits
        epochs = 5
        temp_results = [] 
        for subject_split in range(4):
            trial_idx = np.arange(len(subject_X))
            train_trials, test_trials = nndata.split_idx(subject_split, 4, trial_idx)

            model.load_weights(weights_file % (split))
            h = model.fit(
                subject_X[train_trials,:], subject_y[train_trials,:], 
                validation_data=(subject_X[test_trials,:], subject_y[test_trials,:]),
                epochs=5, batch_size=2, verbose=0
            )
            # save best validation accuracy
            temp_results.append(np.max(h.history["val_acc"]))
        results[current_subject, 1] = np.mean(temp_results)
        

with open('result2.pkl', 'wb') as f:
   pickle.dump(results, f)
        
print("****************************************************************************************************")
        
BATCH = 16
SPLITS = 5
electrodes = range(64)
classes = 3
X,y = nndata.load_raw_data(electrodes, num_classes=classes)

epochs = 3
splits = range(5)
steps_per_epoch = np.prod(X.shape[:2]) / BATCH * (1-1./SPLITS)

#lengths = np.arange(30, 160, 10) # short, less spacing
lengths = np.arange(160, 960, 40) # long, more spacing

results = np.zeros((len(lengths), len(splits), 4, epochs))
for ii,i in enumerate(splits):
    for j, length in enumerate(lengths):
        print("Length %d, SPLIT %d" % (length, i))
        idx = np.arange(len(X))
        train_idx, test_idx = nndata.split_idx(i, 5, idx)
        model = models.create_raw_model(
            nchan=len(electrodes), 
            nclasses=classes, 
            trial_length=length
        )
        
        h = models.fit_model(
            model, X, y, train_idx, test_idx, 
            input_length=length, batch_size=BATCH, 
            steps_per_epoch=steps_per_epoch, epochs=epochs
        )
        
        results[j, ii, :, :] = [ 
            h.history["acc"], 
            h.history["loss"], 
            h.history["val_acc"], 
            h.history["val_loss"]
        ]


with open('result3.pkl', 'wb') as f:
   pickle.dump(results, f)

print("****************************************************************************************************")

classes = 3
split = 0

step = 20
trial_length = 3*160
channels =  range(64)
# load model
model = models.create_raw_model(nchan=len(channels), nclasses=classes, trial_length=input_length)
model.load_weights("weights-%dcl-%d.hdf5" % (classes, split))

idx = range(len(X))
train_idx, test_idx = nndata.split_idx(split, 5, idx)
acc = np.zeros((trial_length-input_length)/ step + 1)
cert = np.zeros((trial_length-input_length)/ step + 1)
falsenull = np.zeros((trial_length-input_length)/ step + 1)
for i, off in enumerate(np.arange(0, trial_length-input_length, step)):
    print(off) 
    xt,yt = nndata.crossval_test(X, y, test_idx, seg_length=input_length, fix_offset=off)
    yp = model.predict(xt)
    cert[i] = yp[yt>0].mean()
    acc[i] = sum(yp.argmax(1)==yt.argmax(1)) / float(len(yt))
    if classes > 2:
        falsenull[i] = sum(yp[yt[:,2]==0].argmax(1)==2) / float(len(yt))
