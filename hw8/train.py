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
from keras.layers import Dropout,Activation, Flatten, Dense, GlobalAveragePooling2D,GlobalMaxPooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
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
batch_size = 32
image_size=(224,224)
input_shape=(224,224, 3)
#otherwise gives an error due to version of keras
os.environ['CUDA_VISIBLE_DEVICES']=''


#set seed
np.random.seed(677)
python_random.seed(677)
tf.random.set_seed(677)



#afs paths
traindir=str(sys.argv[1])
modelname=str(sys.argv[2])






#data generators
#datagen=ImageDataGenerator(validation_split=0.5,rescale=1./255)#, featurewise_center=True)
datagen=ImageDataGenerator(validation_split=0.2,rescale=1./255, featurewise_center=True,featurewise_std_normalization=True,rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,brightness_range=[0.5,1.5],zoom_range=[0.8,1.0],shear_range=0.2)
#datagen=ImageDataGenerator(validation_split=0.5,rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)

traingen=datagen.flow_from_directory(traindir,subset='training', batch_size=batch_size,target_size=image_size,class_mode="categorical")

#if data is fruits we can work with a smaller image size
if len(set(traingen.classes))==101:
    image_size=(128,128)
    input_shape=(128,128, 3)
    batch_size=64
    datagen=ImageDataGenerator(validation_split=0.2,rescale=1./255, featurewise_center=True)
    traingen=datagen.flow_from_directory(traindir,subset='training', batch_size=batch_size,target_size=image_size,class_mode="categorical")
#if dataset is flowers slightly modify the datagenerator
if len(set(traingen.classes))==5:
    image_size=(224,224)
    input_shape=(224,224, 3)
    datagen=ImageDataGenerator(validation_split=0.2,rescale=1./255, featurewise_center=True)
    datagen=ImageDataGenerator(validation_split=0.2,rescale=1./255, featurewise_center=True,featurewise_std_normalization=True,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,brightness_range=[0.5,1.5],zoom_range=[0.7,1.3],shear_range=0.2)
    traingen=datagen.flow_from_directory(traindir,subset='training', batch_size=batch_size,target_size=image_size,class_mode="categorical")

validgen=datagen.flow_from_directory(traindir,subset='validation', batch_size=batch_size,target_size=image_size,class_mode="categorical")
steps_for_each_epoch=int(traingen.n/batch_size)
steps_for_each_epoch_valid=int(validgen.n/batch_size)
print(traingen.classes)
print(len(set(traingen.classes)))

#try several models
#basemodel = applications.Xception(weights = "imagenet", include_top=False, input_shape = input_shape)
#basemodel = applications.NASNetLarge(weights = "imagenet", include_top=False, input_shape = input_shape)
#basemodel = applications.EfficientNetB7(weights = "imagenet", include_top=False, input_shape = input_shape)
basemodel = applications.VGG19(weights = "imagenet", include_top=False, input_shape = input_shape)
#basemodel = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = input_shape)
#basemodel = applications.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = input_shape)

#print("Number of layers in the base model: ", len(basemodel.layers))

model = Sequential()
model.add(basemodel)

if len(set(traingen.classes))==2:
    for layer in basemodel.layers[:19]:
        layer.trainable =  False
    model.add(Flatten())
elif len(set(traingen.classes))==5:
    basemodel = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = input_shape)
    model = Sequential()
    model.add(basemodel)
    for layer in basemodel.layers[:155]:
        layer.trainable =  False
    model.add(Flatten())
else:
    basemodel.trainable=False
    model.add(GlobalMaxPooling2D())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(traingen.classes))))
model.add(Activation('softmax'))







model.summary()


#different optimizers
opt=keras.optimizers.Adam()
#opt=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
if len(set(traingen.classes))==5:
    opt=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)

#model compile
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


#early stoppping condition if val_loss does not improve in 20 epochs
es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,patience=4)
#save the best model that gives lowest val_loss
mc = keras.callbacks.ModelCheckpoint(modelname, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=False)

if len(set(traingen.classes))==5:
    mc = keras.callbacks.ModelCheckpoint(modelname, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True,save_weights_only=False)

#reduce lr by a factor of 0.5 if does not improve for 4 epochs
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3,verbose=1, min_lr=0.0000001,cooldown=1)
#reduce_lr=keras.callbacks.LearningRateScheduler(schedulelr, verbose=1)
#reduce_lr=keras.callbacks.LearningRateScheduler(schedulelr2, verbose=1)
# fit the keras model on the dataset
fit=model.fit_generator(traingen,steps_per_epoch=steps_for_each_epoch,epochs=epochs,validation_data=validgen,callbacks=[es,mc,reduce_lr],validation_steps=steps_for_each_epoch_valid,verbose=1)
#fit=model.fit(train,labels,epochs=epochs,batch_size=batch_size,validation_split=0.2,callbacks=[es,mc,reduce_lr],verbose=1)



