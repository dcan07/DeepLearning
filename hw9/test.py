import keras
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''


modelpath=str(sys.argv[1])
imagedirectory=str(sys.argv[2])


#load model
model = keras.models.load_model(modelpath)
#Noise
rand_w= np.random.normal(0,1, [100, 100] ) 

#use generator
images = model.predict(rand_w)
images = images.reshape(100,28,28)
for i in range(len(images)):
    plt.imshow(images[i], interpolation='nearest')
    plt.axis('off')
    plt.savefig(imagedirectory+'/gan_image'+str(i)+'.png')



