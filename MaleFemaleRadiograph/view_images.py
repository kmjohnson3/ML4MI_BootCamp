#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('echo "Copying Data Locally (Male/Female Radiograph)"')
get_ipython().system('tar xf "/content/drive/My Drive/ML4MI_BOOTCAMP_DATA/MaleFemaleRadiograph.tar" --directory /home/')


# In[ ]:


import os
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model 
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


data_home_dir = '/home/MaleFemaleRadiograph/data/'
train_dir = data_home_dir + 'train'


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator =      train_datagen.flow_from_directory(train_dir, batch_size=30, target_size=(256,256), class_mode='binary', color_mode='grayscale')


# In[ ]:


X,Y = train_generator.next()


# In[ ]:


ind_male = np.array(np.where(Y==1))
ind_female = np.array(np.where(Y==0))


# In[ ]:


pltM = X[ind_male[0,-1],:,:,0]
pltF = X[ind_female[0,-1],:,:,0]


# In[ ]:


m=0
f=0

for i in range(30):
    if Y[i] == 1. and m < 9:
        pltM = np.concatenate((pltM, X[i,:,:,0]),1)
        m+=1
    elif Y[i] == 0. and f < 9:
        pltF = np.concatenate((pltF, X[i,:,:,0]),1)
        f+=1


# In[ ]:


plt.figure(figsize=(20,20))       
plt.imshow(pltF)
plt.figure(figsize=(20,20))
plt.imshow(pltM)     


# In[ ]:


pltM.shape


# In[ ]:




