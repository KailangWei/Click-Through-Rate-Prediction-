# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:03:29 2018

@author: GEASTON
"""

#%% Get Data

import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from keras import backend as K
from keras import optimizers

NEpochs = 1000
BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.01)

def SetTheSeed(Seed):
    np.random.seed(Seed)
    rn.seed(Seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    tf.set_random_seed(Seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

# Read in the data
    
TrainData = pd.read_csv('C:/Users/14702/Downloads/TrainDataForBCExample.csv',sep=',',header=0,quotechar='"')
ValData = pd.read_csv('C:/Users/14702/Downloads/ValDataForBCExample.csv',sep=',',header=0,quotechar='"')

# drop the column not needed 
# TrainData.drop(0,axis = 1) 
# Val_Data = ValData.drop([0,2],axis = 1)

Vars = list(TrainData)
# YTr = np.array(TrainData[[1]])
YTr = np.array(TrainData[Vars[0]])
XTr = np.array(TrainData.loc[:,Vars[1:]])

Vars = list(ValData)
YVal = np.array(ValData[Vars[0]])
XVal = np.array(ValData.loc[:,Vars[1:]])

XTr.shape
XVal.shape

print(YVal)
print(YTr)
min(YTr)
# Rescale Training Data

XTrRsc = (XTr - XTr.min(axis=0))/XTr.ptp(axis=0)
XTrRsc.shape
XTrRsc.min(axis=0)
XTrRsc.max(axis=0)

# Note YTr does not need to be rescaled since it is binary

#Rescale Validation Data. Really should use Training parameters to rescale.
XValRsc = (XVal - XTr.min(axis=0))/XTr.ptp(axis=0)
XValRsc.shape
XValRsc.min(axis=0)
XValRsc.max(axis=0)

# YVal does not need to be rescaled as it is binary.


#%% Set up Neural Net Model - 4 layes of 4 neurons probability output.

#SetTheSeed(3456)

from keras.models import Sequential
from keras.layers import Dense, Activation

BCNN = Sequential()

BCNN.add(Dense(units=4,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNN.add(Dense(units=4,activation="relu",use_bias=True))
BCNN.add(Dense(units=4,activation="relu",use_bias=True))
BCNN.add(Dense(units=4,activation="relu",use_bias=True))
BCNN.add(Dense(units=1,activation="sigmoid",use_bias=True))

BCNN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy'])

#%% Fit NN Model

FitHist = BCNN.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=0)
print("Number of Epochs = "+str(len(FitHist.history['binary_crossentropy'])))
FitHist.history['binary_crossentropy'][-1]
FitHist.history['binary_crossentropy'][-10:-1]

#%% Make Predictions
YHatTr = BCNN.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo.
YHatTr = YHatTr.reshape((YHatTr.shape[0]),)

YHatVal = BCNN.predict(XValRsc,batch_size=XValRsc.shape[0])
YHatVal = YHatVal.reshape((YHatVal.shape[0]),)

#%% Now try using softmax

#SetTheSeed(3456)

from keras.models import Sequential
from keras.layers import Dense, Activation

BCNNsm = Sequential()

BCNNsm.add(Dense(units=4,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=2,activation="softmax",use_bias=True))

BCNNsm.compile(loss='categorical_crossentropy', optimizer=Optimizer,metrics=['categorical_crossentropy'])

#%% Fit NN Model with Softmax

# Need to make YTr an n by 2 matrix

YTr = np.array([1-YTr,YTr]).transpose()

FitHist = BCNNsm.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=0)
print("Number of Epochs = "+str(len(FitHist.history['categorical_crossentropy'])))
FitHist.history['categorical_crossentropy'][-1]
FitHist.history['categorical_crossentropy'][-10:-1]

#%% Make Predictions
YHatTrSM = BCNNsm.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo.
YHatValSM = BCNNsm.predict(XValRsc,batch_size=XValRsc.shape[0]) # Note: Not scaled, so not necessary to undo.

TrOutDF = pd.DataFrame(data={ 'YHatTr': YHatTr, 'YHatTrSM': YHatTrSM[:,1] })
ValOutDF = pd.DataFrame(data={ 'YHatVal': YHatVal, 'YHatValSM': YHatValSM[:,1] })

TrOutDF = pd.DataFrame(data={ 'YHatTr': YHatTr })
ValOutDF = pd.DataFrame(data={ 'YHatVal': YHatVal })

TrOutDF = pd.DataFrame(data={ 'YHatTrSM': YHatTrSM[:,1] })
ValOutDF = pd.DataFrame(data={ 'YHatValSM': YHatValSM[:,1] })


TrOutDF.to_csv('C:/Users/Administrator/Downloads/TrYHatFromBCNN.csv',sep=',',na_rep="NA",header=True,index=False)
ValOutDF.to_csv('C:/Users/Administrator/Downloads/ValYHatFromBCNN.csv',sep=',',na_rep="NA",header=True,index=False)

TrOutDF.to_csv('C:/Users/14702/Downloads/TrYHatFromBCNNsm.csv',sep=',',na_rep="NA",header=True,index=False)
ValOutDF.to_csv('C:/Users/14702/Downloads/ValYHatFromBCNNsm.csv',sep=',',na_rep="NA",header=True,index=False)
