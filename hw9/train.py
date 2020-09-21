import numpy as np

import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input
from keras.datasets import mnist
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''

modelpath=str(sys.argv[1])


def opt():
    return keras.optimizers.Adam(lr=0.0002, beta_1=0.5)


def create_generator():
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=784, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=opt())
    return generator


def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=opt())
    return discriminator



def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=opt())
    return gan




epochs=400
batch=32
#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#standardize the data
x_train=(x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(len(x_train), len(x_train[0,:,:])*len(x_train[0,0,:]))
number_of_batches = int(len(x_train) / batch)
# models
generator_model= create_generator()
discriminator_model= create_discriminator()
gan_model = create_gan(discriminator_model, generator_model)

#holds the values of the loss function
dis_loss,gen_loss=list(), list()


    
for epoch in range(1,epochs+1 ):
    print('Running epoch'+str(epoch))
    gen_l=0
    dis_l=0
    for _ in range(number_of_batches):
        #Random vector
        rand_w= np.random.normal(0,1, [batch, 100] )       
        #generate fake
        fake_images = generator_model.predict(rand_w)        
        # get real images
        real_images =x_train[np.random.randint(0,len(x_train),size=batch)]
        # concatenate real and fake
        data= np.concatenate([real_images, fake_images])      
        # labels for discrimantor
        labels=np.concatenate([np.ones(batch),np.zeros(batch)])
        #run discriminator
        discriminator_model.trainable=True
        dl=discriminator_model.train_on_batch(data, labels)
            
            
        #Tricking the noised input of the Generator as real data
        rand_w= np.random.normal(0,1, [batch, 100])
        labels = np.ones(batch)
        discriminator_model.trainable=False
        generator_model.trainable=True
        gl=gan_model.train_on_batch(rand_w, labels)
        
        #sum the loss in epoch
        gen_l=gen_l+gl
       	dis_l=dis_l+dl
    #get the average loss per epoch
    dis_loss.append(dis_l/number_of_batches)
    gen_loss.append(gen_l/number_of_batches)
    #save model
    generator_model.save(modelpath)


'''
#print the loss of gen and dis
fig, ax = plt.subplots()
epochs = range(1, len(dis_loss) + 1)
ax.plot(epochs, dis_loss, 'g', label='dis_loss')
ax.plot(epochs, gen_loss, 'y', label='gen_loss')
ax.set(title='loss')
#plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig.savefig("loss.png", dpi=1000)
'''
