#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:22:22 2020

@author: dogacanyilmaz
"""


import numpy as np
import sys
from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score



#AFS Paths
trainpath=sys.argv[1]
testpath=sys.argv[2]
nhidden=int(sys.argv[3])
batchsize=int(sys.argv[4])



stepsize=0.001
epochs=1000

#read data
#I want to both read regular data in the course website
#and also liblinear data as in the description

#read regular formatted data
try:
    f = open(trainpath)
    data = np.loadtxt(f)
    train_data = data[:,1:]
    train_label = data[:,0]
    f = open(testpath)
    data = np.loadtxt(f)
    test_data = data[:,1:]
    test_label = data[:,0]
#read liblinear data
except:
    train_data,train_label,test_data,test_label=load_svmlight_files((trainpath,testpath))
    train_data=train_data.toarray()
    test_data=test_data.toarray()



nrows=len(train_data)
ncols=len(train_data[0])


#we can account for bias by adding a column of 1 to the data
bias=np.ones((nrows,1))
train_data=np.append(train_data,bias,axis=1)
#Also add to test data
bias=np.ones((len(test_data),1))
test_data=np.append(test_data,bias,axis=1)
#we must increase ncols
ncols+=1

#Sample random weights for hidden layer
#following samples from uniform [0,1]
hidden_weights=np.random.rand(nhidden)
print('w=',hidden_weights)

#First sample random weights for input layer
#following samples from uniform [0,1]
input_weights=np.random.rand(nhidden,ncols)
print('W=',input_weights)

#sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def calculate_obj(data,label,layer0,layer1):
    return np.sum(np.square(np.matmul(sigmoid(np.matmul(data,layer0.T)),layer1.T)-label))

    #return np.sum(np.square(np.matmul([-1 if a<0 else 1 for a in np.matmul(data,layer0.T)],layer1.T)-label))
    

def calculate_gradient(dataset,labels,w0,w1):
    #initialize gradients 
    #gradient of input layer w is same dimesion as layer 0
    grad0=np.zeros((len(w0),len(w0[0])))
    #grad1 is for hidden layer
    grad1=np.zeros(len(w1))
    #for each instance of data
    for i in range(len(dataset)):
        hiddenvalues=sigmoid(np.matmul(dataset[i],w0.T))
        sqrtloss=np.dot(hiddenvalues,w1.T)-labels[i]
        #first calculate grad1
        #for each element in grad1
        for nhiddennodes in range(len(grad1)):
            grad1[nhiddennodes]+=(2*sqrtloss*hiddenvalues[nhiddennodes])
    
        #now calculate gradients for input layer
        for nhiddennodes in range(len(grad1)):
            grad0[nhiddennodes]+=(2*sqrtloss*w1[nhiddennodes]*hiddenvalues[nhiddennodes]*(1-hiddenvalues[nhiddennodes])*dataset[i])
    
    return grad0,grad1


#calculate initial objective
obj=calculate_obj(train_data,train_label,input_weights,hidden_weights)
prev_obj=np.inf

#main loop
i=0
while(i<epochs):
      #assign objective value
      prev_obj=obj
      
      #SGD Update
      for j in range(nrows):
          #Get indexes
          indexes=np.random.randint(nrows, size=batchsize)
          #calculate gradients with chosen indexes
          d0,d1=calculate_gradient(train_data[indexes],train_label[indexes],input_weights,hidden_weights)
          #Update weights
          input_weights=input_weights-(stepsize*d0)
          hidden_weights=hidden_weights-(stepsize*d1)
    
      #calculate objective value
      obj=calculate_obj(train_data,train_label,input_weights,hidden_weights)

      
      #print epoch anf loss
      i = i + 1
      print("i=",i,"Objective=",obj)
      
#calculate predictions
predictions=np.zeros(len(train_data))
for i in range(len(test_data)):
    predictions[i]=np.dot(sigmoid(np.matmul(train_data[i],input_weights.T)),hidden_weights)
#label based on sign
predictions=[1 if x>0 else -1 for x in predictions]

#print predictions error and weights
print('Prediction=',predictions)
print('Error=',1-accuracy_score(train_label,predictions))
print('w=',hidden_weights)
for i in range(len(input_weights)):
    print('W['+str(i)+']=',input_weights[i])


