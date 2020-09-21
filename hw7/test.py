
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
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os


epochs =200
batch_size = 128

import keras.backend.tensorflow_backend as tfback
#print("tf.__version__ is", tf.__version__)
#print("tf.keras.__version__ is:", tf.keras.__version__)

os.environ['CUDA_VISIBLE_DEVICES']=''
#set seed
#np.random.seed(677)
#python_random.seed(677)
#tf.random.set_seed(677)



#afs paths
testdir=str(sys.argv[1])
modelname=str(sys.argv[2])

#load datasets



image_size = (256,256)
batch_size = 1

datagen=ImageDataGenerator(rescale=1./255)#, featurewise_center=True)
testgen=datagen.flow_from_directory(testdir,target_size=image_size,batch_size=batch_size,class_mode=None,shuffle=False)


#load model
model = keras.models.load_model(modelname)
#model = keras.models.load_weights(modelname)


#print(testgen.n,testgen.batch_size)


STEP_SIZE_TEST=testgen.n/testgen.batch_size
testgen.reset()
predictions=model.predict_generator(testgen,steps=STEP_SIZE_TEST,verbose=1)

predictions=np.argmax(predictions,axis=1)


#print(testgen.classes)

#print(predictions)

error=0
for i in range(len(predictions)):
	if(testgen.classes[i]!=predictions[i]):
		error=error+1


print(error/len(predictions))



