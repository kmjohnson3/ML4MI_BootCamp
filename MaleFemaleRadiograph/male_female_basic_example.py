#!/usr/bin/env python
# coding: utf-8

# ## Download data from Google Drive to colab environment
# First we need to mount the Google Drive folder into colab. <br>
# Then we copy the data for this exercise to the colab VM and untar it "locally".

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:



get_ipython().system('echo "Copying Data Locally (Male/Female Radiograph)"')
get_ipython().system('tar xf "/content/drive/My Drive/ML4MI_BOOTCAMP_DATA/MaleFemaleRadiograph.tar" --directory /home/')


# ## Setup packages and data
# First import the packages you'll need. From Keras, we'll need an data generator package, layers package, a package containing optimizres, and a package that builds/configures models.

# In[ ]:


import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model 
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define data location and image dimensions. Data is split into train (50%), validate (25%), and test (25%).
# We'll use Kera's ImageDataGenerator method to read in the data. Data (.png files) is sorted into folders with the following structure <br>
# >train/<br>
# &ensp;Class1/<br>
# &ensp;&ensp;xx1.png<br>
# &ensp;&ensp;xx2.png<br>
# &ensp;&ensp;...<br>
# &ensp;Class2/<br>
# &ensp;&ensp;yy1.png<br>
# &ensp;&ensp;yy2.png<br>
# test/<br>
# &ensp;Class1/  ...<br>
# &ensp;Class2/  ...<br>
# validation/<br>
# &ensp;Class1/ ...<br>
# &ensp;Class2/ ...<br>
# 
# We tell Keras where the directories are. It counts the number of subfolders and makes each one a class.

# In[ ]:


data_home_dir = '/home/MaleFemaleRadiograph/data/'
train_dir = data_home_dir + 'train'
validation_dir = data_home_dir + 'validation'
dims = 256


# When we define the ImageDataGenerator object, we tell it to normalize the .png images by the max (255)

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)


# Keras will read the files continuously from disk. We tell it where to read, how many to read at a time, what dimensions to resample the images to, and how many image channels there are. These generators will then generate batches of images. 

# In[ ]:


train_generator =      train_datagen.flow_from_directory(train_dir, batch_size=20, target_size=(dims,dims), class_mode='binary', color_mode='grayscale')
validation_generator = valid_datagen.flow_from_directory(validation_dir,batch_size=20, target_size=(dims,dims), class_mode='binary',color_mode='grayscale')


# 
# ## Build network 
# First part of the graph is the input, which, at this point, we only need to tell it its shape (we'll define where the inputs come from when we build the model later)

# In[ ]:


img_input = layers.Input(shape=(dims,dims,1), dtype='float32')


# Now we build our layers of the network. The format is layer_name(_config_info_)(_input_to_layer_).
# Try a simple layer with 1 convolution, max pooling, and a fully-connected layer (these are _not_ the best parameters).

# In[ ]:


x = layers.Conv2D(15, (3, 3), strides=(4,4), padding='same')(img_input)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), strides=None)(x)
x = layers.Flatten()(x)     #reshape to 1xN 
x = layers.Dense(20, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)    #sigmoid for binary


# ## Configure and train model
# We define our model, define the input(s) and output(s). 

# In[ ]:


model = Model(inputs=img_input, outputs=x)


# We then compile it and determine our loss function, our optimizer, and the metrics we want to calculate. This builds the "graph" of our model and computes the functions needed to train it.

# In[ ]:


model.compile(loss = "binary_crossentropy", optimizer = optimizers.RMSprop(learning_rate=1e-5), metrics=["accuracy"])


# This next steps kicks off the network training. This is where we actually feed the compiled model the data (in batches).

# In[ ]:


history = model.fit(train_generator, steps_per_epoch=130, epochs=15, 
                              validation_data=validation_generator, validation_steps=30)


# ## Evaluate performance
# First, let's calculate the performance on our testing dataset

# In[ ]:


test_dir = data_home_dir + 'test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,batch_size=20, target_size=(dims,dims), class_mode='binary',color_mode='grayscale')

#now evaluate the model using the generator
[test_loss, test_acc] = model.evaluate(test_generator, steps=600/20)
print("Test_acc: "+str(test_acc))


# Plot the results using matplotlib

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs,val_acc,'b', label='Validation acc')
plt.legend()
plt.show()


# In[ ]:




