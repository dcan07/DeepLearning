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

numberofepochs=50
accuracytostoptrain=0.02
bestaccuracy=1

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
testDataMean = np.mean(testData, axis=0)
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
#predict dataD and print initial accuracy
predictions=targetmodel.predict(dataD,verbose=1)
print('Accuracy on the 200 instances: ', calculateAccuracy(predictions,dataDlabels))


#create substitute model
def create_substitute():
    substitute = tf.keras.Sequential()
    substitute.add(Flatten(input_shape=(32,32,3)))
    substitute.add(Dense(100))
    substitute.add(Activation('relu'))
    substitute.add(Dense(100))
    substitute.add(Activation('relu'))
    substitute.add(Dense(2, activation="softmax"))
    adam=tf.keras.optimizers.Adam(learning_rate=0.01)
    substitute.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return substitute

loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label,modeltoattack):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = modeltoattack(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  return gradient.numpy()




targetmodel.summary()



es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.0001,cooldown=1, verbose=0)



for epoch in range(numberofepochs):
    print('running epoch '+str(epoch))
    predictions=targetmodel.predict(dataD,verbose=1)
    
    substitutemodel=create_substitute()
    fit=substitutemodel.fit(dataD,predictions,epochs=2000,batch_size=32,validation_split=0.2,callbacks=[es,reduce_lr],verbose=2)
    
    #get gradients
    gradients=create_adversarial_pattern(tf.cast(dataD, tf.float32),tf.cast(dataDlabels, tf.float32),substitutemodel)

    
    

    adversarial=np.empty(dataD.shape)
    for i in range(len(dataD)):
        #Update  the images
        adversarial[i,:,:,:]=dataD[i,:,:,:]+(np.random.choice([-1,1])*lambdavalue*np.sign(gradients[i,:,:,:]))
   
    
    #merge new adversarial images to old if the size is less than 6400
    if len(dataD)<6400:
        dataD=np.concatenate((dataD,adversarial), axis=0)
        dataDlabels=np.concatenate((dataDlabels,dataDlabels), axis=0)
    #else use adversaries as new data
    else:
        dataD=adversarial
        #get gradients
    #check the accuracy on the all but 200 images
    gradients=create_adversarial_pattern(tf.cast(newtestData, tf.float32),tf.cast(newtestLabels, tf.float32),substitutemodel)
    adversarialnewtestData=np.empty(newtestData.shape)
    for i in range(len(newtestData)):
        #Update  the images
        adversarialnewtestData[i,:,:,:]=newtestData[i,:,:,:]+(epsilon*np.sign(gradients[i,:,:,:]))
    predictionsofRemainingData=targetmodel.predict(adversarialnewtestData)
    print('Accuracy on the test set minus 200 images: ', calculateAccuracy(predictionsofRemainingData,newtestLabels))
    #save model if accuracy has decreased
    if calculateAccuracy(predictionsofRemainingData,newtestLabels)<=bestaccuracy:
        bestaccuracy=calculateAccuracy(predictionsofRemainingData,newtestLabels)
        substitutemodel.save(substitutemodelname)
        print('saving')
    #stop running if accuracy is less than treshold    
    if calculateAccuracy(predictionsofRemainingData,newtestLabels)<accuracytostoptrain:
        break

