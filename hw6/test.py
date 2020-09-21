

import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,LeakyReLU
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
#from tensorflow.keras.layers import BatchNormalization
from keras import backend as K
import numpy as np
import random as python_random

print('tensorflow: %s' % tf.__version__)

import sys

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)



#afs paths
testdir=str(sys.argv[1])
labeldir=str(sys.argv[2])
modelname=str(sys.argv[3])



#load datasets
test=np.load(testdir)
labels=np.load(labeldir)

#load model
model = keras.models.load_model(modelname)


if K.image_data_format() == 'channels_first':
        test = test.reshape(test.shape[0], 3, 112, 112)
else:
     	test = test.reshape(test.shape[0], 112, 112, 3)

#scale to -1 to 1
test=(test*2)-1

predictions = model.predict(test)

misclassification=0
for i in range(len(labels)):
	pred=np.argmax(predictions[i,])
	if(pred!=labels[i]):
		misclassification+=1

print('Misclassification error: ',misclassification/len(labels))




