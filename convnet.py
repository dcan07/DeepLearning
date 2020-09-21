#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:42:21 2020

@author: dogacanyilmaz
"""

import numpy as np
import sys
import os
import pandas as pd
from scipy import signal

########################################
### Read images from train directory ###
#AFS Paths
traindir = sys.argv[1]
testdir = sys.argv[2]
df = pd.read_csv(traindir+'/data.csv')#load images' names and labels
names = df['Name'].values
labels = df['Label'].values

traindata = np.empty((len(labels),3,3), dtype=np.float32)
for i in range(0, len(labels)):
	image_matrix = np.loadtxt(traindir+'/'+names[i])
#	traindata = np.append(traindata, np.array(image_matrix, ndmin=3, dtype=int8), axis=0)
#	traindata[i] = np.array(image_matrix, ndmin=3, dtype=np.int8)
	traindata[i] = image_matrix

print(traindata)
print(labels)

sigmoid = lambda x: 1/(1+np.exp(-x))

##############################
### Initialize all weights ###

c = np.random.rand(2,2)
print("c=",c)


epochs = 1000
eta = .1

###########################
### Calculate objective ###
i=0
objective = 0
for i in range(0, len(labels)):
	print("traindata[i]=",traindata[i])
	hidden_layer = signal.convolve2d(traindata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid")
	print(hidden_layer)
	for j in range(0, 2, 1):
		for k in range(0, 2, 1):
			hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
	output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4
	print("output_layer=",output_layer) 
	objective += (output_layer - labels[i])**2
print("objective=",objective) 


#main loop
n=0
while(n<epochs):
      #assign objective value
      prevobj=objective
    
      #calculate gradients for the conv
      #initialize gradients 0
      dfdc1,dfdc2,dfdc3,dfdc4=0,0,0,0
      for i in range(0, len(labels)):

          #Calculate the sqrt of f as in the notes
          hidden_layer = sigmoid(signal.convolve2d(traindata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid"))
          sqrtf = (((hidden_layer.sum().sum())/4)-labels[i])
          
          #For each c get necessary points into points, calculate the convolution
          #then calculate the gradients as in google drive
          points=np.array([[traindata[i,0,0],traindata[i,0,1]],[traindata[i,1,0],traindata[i,1,1]]])
          temp=sigmoid(signal.convolve2d(traindata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid"))
          dfdc1=dfdc1+(0.5*sqrtf*((temp*(1-temp)*points).sum().sum()))
          
          points=np.array([[traindata[i,0,1],traindata[i,0,2]],[traindata[i,1,1],traindata[i,1,2]]])
          temp=sigmoid(signal.convolve2d(traindata[i],np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid"))
          dfdc2=dfdc2+(0.5*sqrtf*((temp*(1-temp)*points).sum().sum()))
          
          points=np.array([[traindata[i,1,0],traindata[i,1,1]],[traindata[i,2,0],traindata[i,2,1]]])
          temp=sigmoid(signal.convolve2d(traindata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid"))
          dfdc3=dfdc3+(0.5*sqrtf*((temp*(1-temp)*points).sum().sum()))
          
          points=np.array([[traindata[i,1,1],traindata[i,1,2]],[traindata[i,2,1],traindata[i,2,2]]])
          temp=sigmoid(signal.convolve2d(traindata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid"))
          dfdc4=dfdc4+(0.5*sqrtf*((temp*(1-temp)*points).sum().sum()))
      
      #Update weights
      c[0,0]=c[0,0]-(eta*dfdc1)
      c[0,1]=c[0,1]-(eta*dfdc2)
      c[1,0]=c[1,0]-(eta*dfdc3)
      c[1,1]=c[1,1]-(eta*dfdc4)

    
      #calculate objective value
      objective = 0
      for i in range(0, len(labels)):
          hidden_layer = sigmoid(signal.convolve2d(traindata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid"))
          output_layer = ((hidden_layer.sum().sum())/4)
          objective += (output_layer - labels[i])**2
      

      #print loss
      print("Objective=",objective)
      n = n + 1


# Read and print test data
df = pd.read_csv(testdir+'/data.csv')#load images' names and labels
testnames = df['Name'].values
testlabels = df['Label'].values

testdata = np.empty((len(testlabels),3,3), dtype=np.float32)
for i in range(0, len(testlabels)):
	image_matrix = np.loadtxt(testdir+'/'+testnames[i])
#	traindata = np.append(traindata, np.array(image_matrix, ndmin=3, dtype=int8), axis=0)
#	traindata[i] = np.array(image_matrix, ndmin=3, dtype=np.int8)
	testdata[i] = image_matrix
print(testdata)
print(testlabels)


print('FINAL')
print("c=",c)

#also calculate accuracy to test
#acc=np.zeros(len(testlabels))



objective = 0
for i in range(0, len(testlabels)):
	print("testdata[i]=",testdata[i])
	hidden_layer = signal.convolve2d(testdata[i], np.array([[c[1,1],c[1,0]],[c[0,1],c[0,0]]]), mode="valid")
	#print(hidden_layer)
	for j in range(0, 2, 1):
		for k in range(0, 2, 1):
			hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
	output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4
	print("output_layer=",output_layer) 
	objective += (output_layer - testlabels[i])**2
	#if output_layer>0.5:
	#	acc[i]=1
print("objective=",objective) 

'''
score=0
for i in range(len(acc)):
    if acc[i]==testlabels[i]:
        score+=1
print('accuracy: ',score/len(testlabels))
'''


