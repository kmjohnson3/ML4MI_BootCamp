#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('echo "Copying Data Locally (Age Regression)"')
get_ipython().system('tar xf "/content/drive/My Drive/ML4MI_BOOTCAMP_DATA/AgeRegressionChallenge.tar" --directory /home/')


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import h5py
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


# Load training, validation, and testing data.
# Convert to nummpy array, add singleton dimension in channel position (1 channel -- grayscale). Edit path as needed.

# In[ ]:


datapath = '/home/AgeRegressionChallenge/Data/Pneumothorax.h5'

with h5py.File(datapath,'r') as f:
    X_test = np.array(f.get('input_test')).astype(np.float32)[:,:,:,np.newaxis]
    Y_test = np.array(f.get('target_test')).astype(np.float32)[:,np.newaxis]   
    X_train = np.array(f.get('input_train')).astype(np.float32)[:,:,:,np.newaxis]  
    Y_train = np.array(f.get('target_train')).astype(np.float32)[:,np.newaxis]   
    X_val =  np.array(f.get('input_val')).astype(np.float32)[:,:,:,np.newaxis]   
    Y_val = np.array(f.get('target_val')).astype(np.float32)[:,np.newaxis]   


# I'll start your network, you build the rest:

# In[ ]:


img_input = Input(shape=(256, 256, 1))


# ## <font color='red'>Enter your model below.</font> 
# Hint: the final layer should have linear activation with 1 output

# In[ ]:


x = ?(img_input)


# Create model.

# In[ ]:


model = Model(inputs=img_input, outputs=x)    


# ## <font color='red'>Compile your model</font> 
# Use the "mean_squared_error" loss function (or try something different! Look up Keras loss functions on Google). Monitor the "mse" metric.

# In[ ]:


model.compile(?)


# Fit the model. Modify the epochs/batch_size as needed. 

# In[ ]:


history = model.fit(x=X_train, y=Y_train, batch_size=15, epochs=50,
                   validation_data = (X_val, Y_val), shuffle=True)


# Plot the training/validation loss

# In[ ]:


mse = history.history['mse']
val_mse = history.history['val_mse']   #validation
epochs = range(1,len(mse)+1)
plt.plot(epochs,mse,'bo', label='Training mse')
plt.plot(epochs,val_mse,'b', label='Validation mse')
plt.legend()
plt.show()


# This is the code to use to evaluate your network -- don't change it. Take a screen shot of the output to submit to the competition! (everyone should submit)

# In[ ]:


Y_pred = model.predict(X_test, batch_size=30)   
Y_pred = np.squeeze(Y_pred)  #remove the singleton dimension for analysis
Y_test = np.squeeze(Y_test)  
plt.scatter(Y_test, Y_pred, s=2)
plt.xlabel('True age')
plt.ylabel('Predicted age')
plt.show()
corr = np.corrcoef(Y_pred, Y_test)   #get correlation matrix
print("Correlation coefficient: " + str(corr[0,1]))


# In[ ]:




