import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
import random as python_random
import sys
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
from sklearn.metrics import accuracy_score
import matplotlib


#calculate accuracy from 2dimensional np array
def calculateAccuracy(ytrue,ypred):
    return accuracy_score(np.argmax(ytrue,axis=1),np.argmax(ypred,axis=1))


epsilon=0.0625


os.environ['CUDA_VISIBLE_DEVICES']=''
#set seed
np.random.seed(677)
python_random.seed(677)
tf.random.set_seed(677)



#paths
testdata=str(sys.argv[1])
testlabel=str(sys.argv[2])
modelname=str(sys.argv[3])

#load datasets
testData = np.load(testdata)
#testData = np.float32(testData)
testLabels = np.load(testlabel)

testData = testData[np.logical_or(testLabels == 0, testLabels == 1)]
testLabels = testLabels[np.logical_or(testLabels == 0, testLabels == 1)]
testLabels = keras.utils.to_categorical(testLabels, 2)

testDataMean = np.mean(testData, axis=0)

#testData = testData - testDataMean
testData=testData/255

#load model
model = keras.models.load_model(modelname)
model.trainable=False


model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])


model.summary()
predictions=model.predict(testData,verbose=2)
accuracy=calculateAccuracy(predictions,testLabels)
print('Accuracy: ',accuracy)

loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label,modeltoattack):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = modeltoattack(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  #signed_grad = tf.sign(gradient)
  #return signed_grad.numpy()
  return gradient.numpy()

#get gradients
gradients=create_adversarial_pattern(tf.cast(testData, tf.float32),tf.cast(testLabels, tf.float32),model)


newtestData=np.empty(testData.shape)
for i in range(len(testData)):
    #Update  the images
    newtestData[i,:,:,:]=testData[i,:,:,:]+(epsilon*np.sign(gradients[i,:,:,:]))



predictions=model.predict(newtestData,verbose=2)
accuracy=calculateAccuracy(predictions,testLabels)
print('Accuracy after white box attack: ',accuracy)

