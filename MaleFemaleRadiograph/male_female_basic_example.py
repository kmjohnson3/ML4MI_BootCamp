
# coding: utf-8

# [View in Colaboratory](https://colab.research.google.com/github/kmjohnson3/ML4MI_BootCamp/blob/master/MaleFemaleRadiograph/male_female_basic_example.ipynb)

# ## Setup packages and data
# First import the packages you'll need. From Keras, we'll need an data generator package, layers package, a package containing optimizres, and a package that builds/configures models.

# In[ ]:


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


# In[ ]:


from keras import optimizers
from keras.models import Model 
from keras import layers
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# This is the command to get the data for linux. For windows please manually install and place "data" in the same folder
get_ipython().system(u'wget https://uwmadison.box.com/shared/static/jjg456te6od73pct27jj8sxn12gpi433.zip')
get_ipython().system(u'unzip -q -o jjg456te6od73pct27jj8sxn12gpi433.zip  ')
  


# Define data location and image dimensions. Data is split into train (50%), validate (25%), and test (25%).
# We'll use Kera's ImageDataGenerator method to read in the data. Data (.png files) is sorted into folders with the following structure
# train/
#     Class1/
#             xx1.png
#             xx2.png
#             ...
#     Class2/
#             yy1.png
#             yy2.png
# test/
#     Class1/  ...
#     Class2/  ...
# validation/
#     Class1/ ...
#     Class2/ ...
# 
# We tell Keras where the directories are. It counts the number of subfolders and makes each one a class.

# In[ ]:


data_home_dir = 'data/'
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


img_input = layers.Input(shape=(dims,dims,1))


# Now we build our layers of the network. The format is layer_name(_config_info_)(_input_to_layer_).
# Try a simple layer with 1 convolution, max pooling, and a fully-connected layer (these are _not_ the best parameters).

# In[ ]:


x = layers.Conv2D(15, (3, 3), strides=(4,4), padding='same', kernel_initializer='he_normal')(img_input)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), strides=None)(x)
x = layers.Flatten()(x)     #reshape to 1xN
x = layers.Dense(20, activation='relu', kernel_initializer='he_normal')(x)
x = layers.Dense(1, activation='sigmoid')(x)    #sigmoid for binary


# ## Configure and train model
# We define our model, define the input(s) and output(s). 

# In[ ]:


model = Model(inputs=img_input, outputs=x)


# We then compile it and determine our loss function, our optimizer, and the metrics we want to calculate. This builds the "graph" of our model and computes the functions needed to train it.

# In[ ]:


model.compile(loss = "binary_crossentropy", optimizer = optimizers.RMSprop(lr=1e-5), metrics=["acc"])


# This next steps kicks off the network training. This is where we actually feed the compiled model the data (in batches).

# In[ ]:


history = model.fit_generator(train_generator, steps_per_epoch=130, epochs=35, 
                              validation_data=validation_generator, validation_steps=30)


# ## Evaluate performance
# First, let's calculate the performance on our testing dataset

# In[ ]:


test_dir = data_home_dir + 'test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,batch_size=20, target_size=(dims,dims), class_mode='binary',color_mode='grayscale')

#now evaluate the model using the generator
[test_loss, test_acc] = model.evaluate_generator(test_generator, steps=600/20)
print("Test_acc: "+str(test_acc))


# Plot the results using matplotlib

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs,val_acc,'b', label='Validation acc')
plt.legend()
plt.show()

