import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import random as python_random
#print('tensorflow: %s' % tf.__version__)
import sys
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout,Activation, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,LeakyReLU,GlobalMaxPooling2D,AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
import os
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
batch_size = 16
image_size = (256,256)
input_shape = (256,256,3)
import keras.backend.tensorflow_backend as tfback
#print("tf.__version__ is", tf.__version__)
#print("tf.keras.__version__ is:", tf.keras.__version__)

#otherwise gives an error due to version of keras
os.environ['CUDA_VISIBLE_DEVICES']=''
#set seed
#np.random.seed(677)
#python_random.seed(677)
#tf.random.set_seed(677)



#afs paths
traindir=str(sys.argv[1])
modelname=str(sys.argv[2])

#load datasets


#data generators
datagen=ImageDataGenerator(validation_split=0.2,rescale=1./255)#,rotation_range=90,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2)
traingen=datagen.flow_from_directory(traindir,subset='training', batch_size=batch_size,target_size=image_size,class_mode="categorical")
validgen=datagen.flow_from_directory(traindir,subset='validation', batch_size=batch_size,target_size=image_size,class_mode="categorical")
steps_for_each_epoch=int(traingen.n/batch_size)



model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

'''
model.add(Conv2D(256, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

'''

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
#opt=keras.optimizers.Adam()
opt=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

#model compile
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


#early stoppping condition if val_loss does not improve in 20 epochs
es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)
#save the best model that gives lowest val_loss
mc = keras.callbacks.ModelCheckpoint(modelname, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=False)

#reduce lr by a factor of 0.5 if does not improve for 4 epochs
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3,verbose=1, min_lr=0.0000001,cooldown=1)
# fit the keras model on the dataset
fit=model.fit_generator(traingen,steps_per_epoch=steps_for_each_epoch,epochs=epochs,validation_data=validgen,callbacks=[es,mc,reduce_lr],validation_steps=78,verbose=1)
