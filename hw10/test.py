import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#from keras import backend as K
import random as python_random
import sys
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import Callback



#calculate accuracy from 2dimensional np array
def calculateAccuracy(ytrue,ypred):
    return accuracy_score(np.argmax(ytrue,axis=1),np.argmax(ypred,axis=1))

epsilon=0.0625
lambdavalue=0.1
os.environ['CUDA_VISIBLE_DEVICES']=''
#set seed
np.random.seed(677)
python_random.seed(677)
tf.random.set_seed(677)



#paths
testdata=str(sys.argv[1])
testlabel=str(sys.argv[2])
targetmodelname=str(sys.argv[3])
substitutemodelname=str(sys.argv[4])

if not substitutemodelname.endswith(".hdf5"):
    substitutemodelname=substitutemodelname+str(".hdf5")

#load datasets
testData = np.load(testdata)
testData = np.float32(testData)
testLabels = np.load(testlabel)

testData = testData[np.logical_or(testLabels == 0, testLabels == 1)]
testLabels = testLabels[np.logical_or(testLabels == 0, testLabels == 1)]
testLabels = tf.keras.utils.to_categorical(testLabels, 2)
#divide by mean
testData = testData/255



#get 200 of the test data
indexes=np.random.choice(len(testData), size=200, replace=False)
dataD=testData[indexes,:,:,:]
dataDlabels=testLabels[indexes,:]

#And the remaining of the test data
newtestData=np.delete(testData,indexes,axis=0)
newtestLabels=np.delete(testLabels,indexes,axis=0)

#load target model
targetmodel = tf.keras.models.load_model(targetmodelname)

predictionsoftestDataBeforeAdversaries=targetmodel.predict(testData,verbose=0)
print('Accuracy before creating adversaries: ', calculateAccuracy(predictionsoftestDataBeforeAdversaries,testLabels))



loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label,modeltoattack):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = modeltoattack(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  return gradient.numpy()


#load substitute model
substitutemodel = tf.keras.models.load_model(substitutemodelname)

#get gradients
gradients=create_adversarial_pattern(tf.cast(testData, tf.float32),tf.cast(testLabels, tf.float32),substitutemodel)
adversarialtestData=np.empty(testData.shape)
for i in range(len(testData)):
    #Update  the images
    adversarialtestData[i,:,:,:]=testData[i,:,:,:]+(epsilon*np.sign(gradients[i,:,:,:]))
#print accuracy
predictionsoftestDataAfterAdversaries=targetmodel.predict(adversarialtestData)
print('Accuracy after creating adversaries: ', calculateAccuracy(predictionsoftestDataAfterAdversaries,testLabels))

