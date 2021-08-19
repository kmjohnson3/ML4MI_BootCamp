#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kmjohnson3/ML4MI_Bootcamp_Development/blob/master/FunctionFitting/FunctionFitter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # This is code to do very simple regression of functions 
# Initially this is set to fit x1+x2 using a single 2 neuron hidden layer. This code makes a data set simply by evaulating the function. Some excersises:
# 
# 1. Verify the number of network parameters match the expected. 
# 2. Change the function to a polynomial (a*x1^2+b*x2+c) or other function. Does it still fit well?
# 3. Change the network to improve the fit of your function in (2)
# 4. Add additional layers and evaluate the fit (2)
# 5. Try other more complex functions

# In[ ]:


#Import some libraries, some are used, some could be!

# Core libraries
import tensorflow.keras as keras 
from tensorflow.keras import Input, Model

# Keras Layers for Networks
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization

# Numpy array library
import numpy as np


# In[ ]:


'''
Generate random data of two inputs. 
'''

# Simple random data (2*X1 + X2), size is (number of examples,outputs,inputs)
x = np.random.randn(100000,1,2)

# Define as function so you only have to change once in the code
def target_function (x1,x2):
    return(2*x1 + x2 )

# Evaluate target function
y = target_function (x[:,:,0],x[:,:,1])
y = np.expand_dims(y,axis=2)

print('Shape of output ' + str(y.shape))
print('Shape of input ' + str(x.shape))


# In[ ]:


# Make the two neuron network described in the powerpoint. Its two inputs, a 2 neuron hidden layer, and and one output layer

# Define input placeholder
i = Input(shape=(1,2))

# create hidden layer (Dense is fully connected in this 1D example)
hidden_layer = keras.layers.Dense(2,activation='relu',use_bias=False)(i)

# create output layer 
o = keras.layers.Dense(1,activation='linear',use_bias=False)(hidden_layer)

# Define the model
model = keras.Model(inputs=i, outputs=o)

# Print the summary
model.summary()    


# In[ ]:


# Fit the  model
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x,y,epochs=10, batch_size=100)


# In[ ]:


# This pulls out the weights from the model
weights = model.get_weights()


# In[ ]:


#Print the layers
print('Layer 1')
print(weights[0])
print('Layer 2')
print(weights[1])


# In[ ]:


# Now test with some random number
test = np.random.randn(1,1,2)
out = model.predict(test)
act = target_function(test[:,:,0],test[:,:,1])
print('A = ' + str(test[0,0,0]) + 'B = ' + str(test[0,0,1]) + ' Predict = ' + str(out[0,0,0]) + '  Actual = ' + str(act[0,0]) )


# In[ ]:


# Plot over a wider range (on a grid this time)
x1, x2 = np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
x1 = np.reshape(x1,(-1,1,1))
x2 = np.reshape(x2,(-1,1,1))
xtest = np.concatenate((x1,x2),axis=2)
ytest = target_function( xtest[:,:,0],xtest[:,:,1])
ytest = np.expand_dims(ytest,2)

# Do the inference ( prediction)
ypredict = model.predict(xtest)

#This imports the plotting tools. First line is to allow interactive on cloud
import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(np.reshape(ytest,(100,100)))
plt.colorbar()
plt.ylabel('X1')
plt.xlabel('X2')
plt.title('True')

plt.subplot(132)
plt.imshow(np.reshape(ypredict,(100,100)))
plt.colorbar()
plt.ylabel('X1')
plt.xlabel('X2')
plt.title('Predicted')

plt.subplot(133)
plt.imshow(np.reshape(ypredict-ytest,(100,100)))
plt.colorbar()
plt.ylabel('X1')
plt.xlabel('X2')
plt.title('Difference')

plt.show()

