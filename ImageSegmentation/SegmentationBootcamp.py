'''Code demonstration for introductory deep learning programming
in Keras. This tutorial walks through an example in segmentation
'''
# for pydicom install: conda install -c conda-forge --no-deps pydicom
# other deps:
    # pip install keras
    # conda install scikit-image
    # conda install scipy
    # Demo_functions.py must be on path
    
#%% Initial Preparation
# First, import necessary modules
import os # operating system operations 
import numpy as np # number crunching
import keras # our deep learning library
import matplotlib.pyplot as plt # for plotting our results

#%% Data preparation
# As with all deep learning applications, we need to do some data
# preparation first. It's generally far more time consuming than the actual
# deep learning part.
# This guide assumes that you have the data downloaded and extracted to
# the sub-directory 'LCTSC' in your working directory

# We have to come up with a way to load in all that data in an organized way.
# First, let's get all the subject directories. We'll do this by proceeding
# through the directory structure and grabbing the ones we want.
# We'll use the package glob to make this easy
import glob
# We know our initial directory: LCTSC. Let's add that to our current
# directory to get the full path
initial_dir = os.path.join(os.getcwd(),'LCTSC')
# Now we'll get all the subject directories using glob
subj_dirs = glob.glob(os.path.join(initial_dir,'LCTSC*'))
# Now all the subject directories are contained in a list
# Let's grab the first one in that list and look for the data
cur_dir = subj_dirs[1]
# The next directory level just has 1 directory, so we'll grab that
cur_dir = glob.glob(os.path.join(cur_dir, "*", ""))[0]
# Now we have the dicom image directory and the label directory
# The dicom iamge directory starts with a 0 so we'll find that one first
dcm_dir = glob.glob(os.path.join(cur_dir, "0*", ""))[0]
# Let's grab the label directory while we're at it. It starts with a 1
lbl_dir = glob.glob(os.path.join(cur_dir, "1*", ""))[0]
# Now, we can get the list of dicom files that we need to load for this subject
# We just have to look for .dcm files in the dcm_dir we found
dicom_files = glob.glob(os.path.join(dcm_dir, "*.dcm"))
# Great. Let's get the label filepath too
# It's just contained in a single dicom-rt file in the label directory
lbl_file = glob.glob(os.path.join(lbl_dir,"*.dcm"))[0]
# Great! We have all the file paths for this subject. Now we need
# to actually load in the data
# We'll need the PyDicom package to read the dicoms
import pydicom
# First, we'll load in all the dicom data to a list
dicms = [pydicom.read_file(fn) for fn in dicom_files]
# These likely won't be in slice order, so we'll need to sort them
dicms.sort(key = lambda x: float(x.ImagePositionPatient[2]))
# Then, stack all the pixel data together into a 3D array
# We'll convert to floats while doing this
ims = np.stack([dcm.pixel_array.astype(np.float) for dcm in dicms])
# The last thing we will do is normalize all the images to [0,1]
# There are a variety of normalization methods used, but
# this is simple and seems to work just fine
for im in ims:
    im /= np.max(im)
    
# Now that we have our inputs, we need targets for our
# deep learning model.
# Let's go back and load the label file we already found
label = pydicom.read_file(lbl_file)
# This gets ugly, but we need to extract the contour data
# stored in the label and convert it to masks that can be fed
# to the deep learning model.
# First, get the contour data. We will focus on the lungs for this tutorial
# We need to figure out which contours are the lungs
contour_names = [s.ROIName for s in label.StructureSetROISequence]
# Get the right and left lung indices
r_ind = contour_names.index('Lung_R')
l_ind = contour_names.index('Lung_L')
# Extract the corresponding contours and combine
contour_right = [s.ContourData for s in label.ROIContourSequence[r_ind].ContourSequence]
contour_left = [s.ContourData for s in label.ROIContourSequence[l_ind].ContourSequence]
contours = contour_left + contour_right
# Next, we need to setup the coordinate system for our images
# to make sure our contours are aligned
# First, the z position
z = [d.ImagePositionPatient[2] for d in dicms]
# Now the rows and columns
# We need both the position of the origin and the
# spacing between voxels
pos_r = dicms[0].ImagePositionPatient[1]
spacing_r = dicms[0].PixelSpacing[1]
pos_c = dicms[0].ImagePositionPatient[0]
spacing_c = dicms[0].PixelSpacing[0]
# Now we are ready to create our mask
# First, preallocate
mask = np.zeros_like(ims)    
# we are going to need a contour-to-mask converter
from skimage.draw import polygon
# now loop over the different slices that each contour is on
for c in contours:
    nodes = np.array(c).reshape((-1, 3))
    assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
    zNew = [round(elem,1) for elem in z]
    try:
        z_index = z.index(nodes[0,2])
    except ValueError:
        z_index = zNew.index(nodes[0,2])
    r = (nodes[:, 1] - pos_r) / spacing_r
    c = (nodes[:, 0] - pos_c) / spacing_c
    rr, cc = polygon(r, c)
    mask[z_index,rr, cc] = 1

# Now we have a mask!
# We have all the pieces we need:
# Inputs, and targets
# Now we just to repeat for all of the subjects
# Luckily, there is a pre-made function that does everything we
# just did for whatever directory we give it. So all we have to do is
# call this function on all the directories we already collected
from Demo_Functions import GetLCTSCdata
data = [GetLCTSCdata(d) for d in subj_dirs]
# get all images together as inputs
inputs = np.concatenate([d[0] for d in data])
# get all masks together as targets
targets = np.concatenate([d[1] for d in data])
# clear a couple large variables that are no longer needed
del data
del ims

# Just a couple more pre-processing steps.
# First, our images are 512x512. That's pretty large for most
# deep learning applications. It's certainly doable, but for the
# purpose of this demonstration we will downsample to
# 256x256 so that the processing is faster
# we'll use another scipy function for this
from scipy.ndimage import zoom
inputs = zoom(inputs, (1,.5,.5))
targets = zoom(targets, (1,.5,.5))
targets[targets>.1] = 1
targets[targets<.1] = 0
# the final step is to add a singleton dimesion to these arrays
# This is necessary because the deep learning model we will create
# will expect our input to have color channels. Since our images
# are grayscale, they will just have a single color channel
inputs = inputs[...,np.newaxis]
targets = targets[...,np.newaxis]

# So far so good! Our data is now ready for training!

# But, wait. We don't just need training data. We need a 
# way of determining if our model is overfitting
# We can split some of our data off and use it for validation
# during the training process
# Let's take 20% of the last slices and use them for this purpose
# This will be equal to the last two subjects, so we won't
# have any overlap of subjects between the different sets
# Get the total number of slices
num_slices = inputs.shape[0]
# Find the cutoff
split_ind = np.int(.8*num_slices)
# split into training and validation using common nomenclature
x_train = inputs[:split_ind]
y_train = targets[:split_ind]
x_val = inputs[split_ind:]
y_val = targets[split_ind:]
# finally, shuffle the order of the training data
# being sure to keep the inputs and targets in the 
# same order...
sort_r = np.random.permutation(split_ind)
np.take(x_train,sort_r,axis=0,out=x_train)
np.take(y_train,sort_r,axis=0,out=y_train)
# clear up unneeded variables
del inputs,targets

#%% Building a segmentation network

# We will build a deep convolutional neural network layer by layer
# We first need an input layer that takes our inputs
from keras.layers import Input
# Our input layer just needs the shape of the input we are providing
# The shape dimensions are sample,row,column,channel
# For this 2D network, our samples are different slices
# We don't need to provide this dimension to the input layer, since
# we will feed those samples in as batches during training. But
# we need the rest of the dimensions
inp = Input(shape=x_train.shape[1:])
# Now, we will add on convolutional layers
from keras.layers import Conv2D
# We can reuse the variable 'x' and Keras will remember what the layers
# are connected to
x = Conv2D(20,(3,3),activation='relu')(inp)
x = Conv2D(40,(3,3),activation='relu')(x)
x = Conv2D(40,(3,3),activation='relu')(x)
# now we will use a strided convolution, which downsamples the input
# and increases the network's receptive field
# We will use zero padding first to make the image shapes work out correctly
from keras.layers import ZeroPadding2D
x = ZeroPadding2D(padding=(1,1))(x)
x = Conv2D(40,(4,4),strides=(2,2),activation='relu')(x)
# repeat that sequence
x = Conv2D(60,(3,3),activation='relu')(x)
x = Conv2D(80,(3,3),activation='relu')(x)
x = Conv2D(80,(3,3),activation='relu')(x)
x = ZeroPadding2D(padding=(1,1))(x)
x = Conv2D(80,(4,4),strides=(2,2),activation='relu')(x)
# now, we will reverse the downsampling using transposed convolutions
from keras.layers import Conv2DTranspose
x = Conv2DTranspose(60,(4,4),strides=(2,2),activation='relu')(x)
x = Conv2DTranspose(40,(3,3),activation='relu')(x)
x = Conv2DTranspose(40,(3,3),activation='relu')(x)
x = Conv2DTranspose(40,(4,4),strides=(2,2),activation='relu')(x)
x = Conv2DTranspose(20,(3,3),activation='relu')(x)
x = Conv2DTranspose(20,(3,3),activation='relu')(x)
x = Conv2DTranspose(10,(3,3),activation='relu')(x)
# finally, our output layer will need to have a single output
# channel corresponding to a single segmentation class
# We will use sigmoid activation that squashed the output to a probability
out = Conv2D(1,(1,1),activation='sigmoid')(x)
# Now, we have a graph of layers created but they are not yet a model
# Fortunately, Keras makes it easy to make a model out of a graph
# just using the input and output layers
from keras.models import Model
SegModel = Model(inp,out)
# We have a deep learning model created! Let's take a look to make
# sure we got the image shapes to work out
SegModel.summary()

#%% Compiling and training the model

# Compiling the model is the final step before it is ready to train.
# We need to define our loss function and optimizer that Keras will 
# use to run the training

# The Dice coefficient is not only a good segmentation metric,
# is also works well as a segmentation loss function since it
# can be converted to being differentiable without much difficulty
# Loss functions in Keras need be defined using tensor functions
# Here is what that looks like:
import keras.backend as K
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    # We have calculated dice, but we want to maximize it. Keras tries to minimize
    # the loss so we simply return 1- dice
    return 1-dice


# The ADAM optimizer is widely used with good performance on the majority
# of deep learning applications
SegModel.compile(loss=dice_coef,optimizer=keras.optimizers.Adam(lr=1e-4))

# All that's left to do is to "fit" the model to our data!
# We supply our training data, our batch size and epochs that were
# defined earlier, we ask Keras to constantly report progress (verbose), 
# and we supply some validation data that will be evaluated at the end
# of every epoch so we can keep an eye on overfitting
SegModel.fit(x_train, y_train,
          batch_size=16,
          epochs=16,
          verbose=1,
          shuffle=True,
          validation_data=(x_val, y_val))
# After the training is complete, we evaluate the model again on our validation
# data to see the results.
score = SegModel.evaluate(x_val, y_val, verbose=0)
print('Final Dice on validation set:', 1-score)


# We'll display the prediction and truth next to each other
# and see how it faired
predictions = SegModel.predict(x_val)
# pick a random slice to examine
disp_ind = 42
plt.figure()
disp = np.c_[x_val[disp_ind,...,0],predictions[disp_ind,...,0],y_val[disp_ind,...,0]]
plt.imshow(disp,cmap='gray')

# Well... it's ok. Not the greatest but it's not terrible either.
# There are a variety of directions to go from here

# A deeper net gives more representational power
# to the model. If the problem is too complex for the current
# network, making it deeper should improve performance

# Some mathematical tricks, like batch normalization and
# ELU activations can help with the learning process
# and make the model learn quicker

# In segmentation, a particularly useful trick is the use of
# skip connetions, in which layers from the downsampling part
# of the network are concatenated with layers on the
# upsampling part. This both boosts the representational power
# of the model as well as improves the gradient flow, which
# also helps the model learn quicker.


#%% Part 3: Functional API
# So far, we've been making sequential models.
# Basically, it means that our network
# has a single, straight path, ie 
# input -> convolution -> activation -> convolution -> output
# Each layer has a single input and output
#
# But what if we wanted something more complicated? What if
# we had two different inputs, for example? Then we would want
# something like
# input1 --> concatenate -> convolution -> ...
# input2 ----^
# Or maybe, what I mentioned as an alteration to the segmentation
# network:
# input -> conv1 -> conv2 -> deconv1 -> concatenate -> deconv2 -> output
#               `----------------------^
# The extra connection shown is called a skip connection.
# Skip connections allow the model to consider features that were calculated
# earlier in the network again, merged with further processed features
# in practice, this has shown to be hugely helpful in geting precise
# localization in segmentation outputs

# We'll use the same segmentation data so no need to prepare anything new.
# Let's jump into model creation.


#%% Build a segmentation model with skip connections


# A new layer we will need for this model
from keras.layers import concatenate

# start like before
inp = Input(shape=x_train.shape[1:])
# add on a couple convolutional layers
# We don't need to keep track of every layer- just
# a few of them. We won't keep track of the first one
# but we'll keep the second one and name it x1
x = Conv2D(20,kernel_size=(3,3),padding='same',activation='relu')(inp)
x1 = Conv2D(40, kernel_size=(3,3),padding='same',activation='relu')(x)
# Add zero padding like before to keep our layer sizes friendly
# and then perform downsampling
zp = ZeroPadding2D(padding=(1,1))(x1)
x = Conv2D(40, kernel_size=(4,4),
                 strides=(2,2),
                 activation='relu')(zp)
# Now repeat the process, hanging onto the second layer again
x = Conv2D(60, kernel_size=(3,3),padding='same',activation='relu')(x)
x2 = Conv2D(60, kernel_size=(3,3),padding='same',activation='relu')(x)
zp = ZeroPadding2D(padding=(1,1))(x2)
x = Conv2D(60, kernel_size=(4,4),
                strides=(2,2),
                activation='relu')(zp)
# We've now done 2 downsampling layers, like before.
# Now for the decoding side of the network, we will start
# adding skip connections
# The first couple of layers are the same as usual.
x = Conv2D(60, kernel_size=(3,3),padding='same',activation='relu')(x)
x = Conv2D(60, kernel_size=(3,3),padding='same',activation='relu')(x)
# Now our upsampling layer
x = Conv2DTranspose(60, kernel_size=(4,4),
                          strides=(2,2),
                          activation='relu')(x)
x = Conv2D(60, kernel_size=(3,3),activation='relu')(x)
# This layer is now the same size as the second layer we kept.
# It can be tough to get layers to match up just right in size
# Playing around with kernel size and strides is usually needed
# so that concatenation can take place. The x,y spatial dimensions
# must be the same. Number of channels doesn't matter.
# Luckily, we already did the work for you so these layers can be
# concatenated
x = concatenate([x,x2])
# Now continue to add layers for the decoding side of the
# network, treating this merged layer like any other
x = Conv2D(40, kernel_size=(3,3),padding='same',activation='relu')(x)
x = Conv2D(40, kernel_size=(3,3),padding='same',activation='relu')(x)
x = Conv2DTranspose(40, kernel_size=(4,4),
                          strides=(2,2),
                          activation='relu')(x)
x = Conv2D(40, kernel_size=(3,3),activation='relu')(x)
x = concatenate([x,x1])
x = Conv2D(20, kernel_size=(3,3),padding='same',activation='relu')(x)
x = Conv2D(20, kernel_size=(3,3),padding='same',activation='relu')(x)

# Final output layer
out = Conv2D(1,kernel_size=(1,1),activation='sigmoid')(x)

SegModel2 = Model(inp,out)

# We can print out a summary of the model to make sure it's what we want.
# It's a little bit harder to keep track of layers in non-sequential
# format, but it's still a good way to make sure things look right.
SegModel2.summary()

#%%
# Now, everything else is just like the previous segmentation model
# Let's try it out and see how it works!
SegModel2.compile(loss=dice_coef,optimizer=keras.optimizers.Adam())
SegModel2.fit(x_train, y_train,
          batch_size=16,
          epochs=10,
          verbose=1,
          shuffle=True,
          validation_data=(x_val, y_val))

predictions = SegModel2.predict(x_val)
plt.figure()
disp_ind = 42
disp = np.c_[x_val[disp_ind,...,0],predictions[disp_ind,...,0],y_val[disp_ind,...,0]]
plt.imshow(disp,cmap='gray')


# Well.... ok. It's about the same.
# However! In the long run (more than 10 epochs), having these skip connections
# will definitely make a difference. The difference becomes more pronounced
# for deeper networks (more layers) with more parameters and larger images.

# Now that you know the functional API, you can make any graph you like, train
# it, and use it! Once you've mastered the syntax and conceptual understanding
# of how to connect layers, you are only limited by your imagination as far
# as what kind of network you can build.

# Best of luck, and happy deep learning!