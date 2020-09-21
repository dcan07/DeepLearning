# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:58:35 2020

@author: dy234
"""


import numpy as np
import tensorflow as tf
import keras
#from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,LeakyReLU,GlobalMaxPooling2D,AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
#from tensorflow.keras.layers import BatchNormalization
from keras import backend as K
import numpy as np
import random as python_random
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.regularizers import l2
#print('tensorflow: %s' % tf.__version__)

import sys

def schedulelr(epoch,learning_rate):
	if epoch%10==0:
		learning_rate=learning_rate/10
		print('new learning rate ',learning_rate)
	return learning_rate

def schedulelr2(epoch,learning_rate):
        if epoch>10:
                learning_rate=learning_rate*1/(1 + 0.002 * epoch)
                print('new learning rate ',learning_rate)
        return learning_rate


epochs =200
batch_size = 64

import keras.backend.tensorflow_backend as tfback
#print("tf.__version__ is", tf.__version__)
#print("tf.keras.__version__ is:", tf.keras.__version__)

#otherwise gives an error due to version of keras
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

#set seed
#np.random.seed(677)
#python_random.seed(677)
#tf.random.set_seed(677)



#afs paths
traindir=str(sys.argv[1])
labeldir=str(sys.argv[2])
modelname=str(sys.argv[3])

#load datasets
train=np.load(traindir)
labels=np.load(labeldir)

#train=train[0:200,:,:,:]
#labels=labels[0:200]

#reshape the data
if K.image_data_format() == 'channels_first':
        train = train.reshape(train.shape[0], 3, 112, 112)
        input_shape = (3, 112, 112)
else:
        train = train.reshape(train.shape[0], 112, 112, 3)
        input_shape = (112, 112, 3)

#print('train shape:', train.shape)
#print(train.shape[0], 'train samples')

# one hor encoding to the labels
temp = np.zeros((labels.size, 10))
temp[np.arange(labels.size),labels] = 1
labels=temp

reg=0.1
#scale to -1 to 1
train=(train*2)-1

model = Sequential()
model.add(Conv2D(256, (3,3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(256, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))



model.add(AveragePooling2D(pool_size=(6,6)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


#different optimizers
#opt=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, amsgrad=False,clipnorm=1,clipvalue=1)
opt=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#opt=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
#opt=keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
#opt = keras.optimizers.SGD(learning_rate=0.001)

#model compile
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


#early stoppping condition if val_loss does not improve in 20 epochs
#es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
#save the best model that gives lowest val_loss
#mc2 = keras.callbacks.ModelCheckpoint(modelname+'2', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=24)
#save the best model that gives lowest val_loss
mc = keras.callbacks.ModelCheckpoint(modelname, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

#terminate on error on loss
nanstop=keras.callbacks.TerminateOnNaN()
#reduce lr by a factor of 0.5 if does not improve for 4 epochs
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,verbose=1, min_lr=0.0000001,cooldown=1)
#reduce_lr=keras.callbacks.LearningRateScheduler(schedulelr, verbose=1)
#reduce_lr=keras.callbacks.LearningRateScheduler(schedulelr2, verbose=1)
# fit the keras model on the dataset
fit=model.fit(train,labels,epochs=epochs,batch_size=batch_size,validation_split=0.01,callbacks=[es,mc,nanstop,reduce_lr],verbose=2)




