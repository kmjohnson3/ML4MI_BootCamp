
# coding: utf-8

# In[ ]:


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import h5py
from keras import optimizers
from keras.models import Model 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense


# Load training, validation, and testing data.
# Convert to nummpy array, add singleton dimension in channel position (1 channel -- grayscale). Edit path as needed.
# Note that a warning may appear about expand_dims using deprecated functions -- ignore it or swap expand_dims with X_test[:,:,:, np.newaxis]

# In[ ]:


datapath = 'Data/Pneumothorax.h5'

with h5py.File(datapath,'r') as f:
    X_test = np.expand_dims( np.array(f.get('input_test')) , 3).astype(np.float32)
    Y_test = np.expand_dims( np.array(f.get('target_test')) , 3).astype(np.float32)   
    X_train = np.expand_dims( np.array(f.get('input_train')) , 3).astype(np.float32)  
    Y_train = np.expand_dims( np.array(f.get('target_train')) , 3).astype(np.float32)   
    X_val =  np.expand_dims( np.array(f.get('input_val')) , 3).astype(np.float32)   
    Y_val = np.expand_dims( np.array(f.get('target_val')) , 3).astype(np.float32)   


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


model.compile()


# Fit the model. Modify the epochs/batch_size as needed. 

# In[ ]:


history = model.fit(x=X_train, y=Y_train, batch_size=15, epochs=50,
                   validation_data = (X_val, Y_val), shuffle=True)


# Plot the training/validation loss

# In[ ]:


mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']   #validation
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

