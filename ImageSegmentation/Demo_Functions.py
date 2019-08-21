#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:32:46 2018

@author: jmj136
"""
import os
import glob
import numpy as np
import pydicom
from skimage.draw import polygon
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from natsort import natsorted
from skimage.transform import resize

def ProcessDicomImage(dcm):
    # convert to numpy array of floats
    array = dcm.pixel_array.astype(np.float32)
    # normalize to mean of 0 and std of .01
#     array -= np.mean(array)
#     array /= 100*np.std(array)
    array -= np.min(array)
    array /= np.max(array)
    # resize to (256,256)
    array = resize(array,(256,256),mode='constant')
    # scale back to std of 1
    return array

def GetLCTSCdata(directory):
    cur_dir = glob.glob(os.path.join(directory, "*", ""))[0]
    dcm_dir = glob.glob(os.path.join(cur_dir, "0*", ""))[0]
    lbl_dir = glob.glob(os.path.join(cur_dir, "1*", ""))[0]
    dicom_files = natsorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    lbl_file = glob.glob(os.path.join(lbl_dir,"*.dcm"))[0]
    dicms = [pydicom.read_file(fn) for fn in dicom_files]
    ims = np.stack([ProcessDicomImage(d) for d in dicms])
    # get labels
    label = pydicom.read_file(lbl_file)
    contour_names = [s.ROIName for s in label.StructureSetROISequence]
    # Get the right and left lung indices
    r_ind = contour_names.index('Lung_R')
    l_ind = contour_names.index('Lung_L')
    # Extract the corresponding contours and combine
    contour_right = [s.ContourData for s in label.ROIContourSequence[r_ind].ContourSequence]
    contour_left = [s.ContourData for s in label.ROIContourSequence[l_ind].ContourSequence]
    contours = contour_left + contour_right
    # Z positions
    z = [d.ImagePositionPatient[2] for d in dicms]
    z_R = [round(elem,1) for elem in z]
    # Rows and columns
    pos_r = dicms[0].ImagePositionPatient[1]
    spacing_r = dicms[0].PixelSpacing[1]
    pos_c = dicms[0].ImagePositionPatient[0]
    spacing_c = dicms[0].PixelSpacing[0]
    # Preallocate
    mask = np.zeros_like(ims)
    # loop over the different slices that each contour is on
    for c in contours:
        tempMask = np.zeros((512,512),dtype=np.float32)
        nodes = np.array(c).reshape((-1, 3))
        assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        try:
            z_index = z.index(nodes[0,2])
        except ValueError:
            z_index = z_R.index(nodes[0,2])
        r = (nodes[:, 1] - pos_r) / spacing_r
        c = (nodes[:, 0] - pos_c) / spacing_c
        rr, cc = polygon(r, c)
        tempMask[rr,cc] = 1
        mask[z_index] += resize(tempMask,(256,256),mode='constant')
    return ims,mask

def GetLungSegData(initial_dir):
    # First, let's get all the subject directories. We'll do this by proceeding
    # through the directory structure and grabbing the ones we want.
    # We'll use the package glob for finding directories
    # The input to this function was the LCTSC directory
    
    # Now we'll get all the subject directories using glob
    subj_dirs = glob.glob(os.path.join(initial_dir,'LCTSC*'))
    # and feed those directories into another function that loads
    # the dicoms and masks for each
    data = [GetLCTSCdata(d) for d in tqdm(subj_dirs,desc='Loading data:')]
    # get all images together as inputs
    inputs = np.concatenate([d[0] for d in data])
    # get all masks together as targets
    targets = np.concatenate([d[1] for d in data])
    # add a singleton dimension to the input and target arrays
    inputs = inputs[...,np.newaxis]
    targets = targets[...,np.newaxis]
    # Get the total number of slices
    num_slices = inputs.shape[0]
    # Find the cutoff- set to 90% train and 10% validation
    split_ind = np.int(.1*num_slices)
    # split into training and validation sets using the cutoff
    x_val = inputs[:split_ind]
    y_val = targets[:split_ind]
    x_train = inputs[split_ind:]
    y_train = targets[split_ind:]
    tqdm.write('Data loaded')
    return x_train,y_train,x_val,y_val

#%%
def display_mask(im,mask,name='Mask Display'):
    msksiz = np.r_[mask.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*mask.astype(float)
    
    im = im-np.min(im)
    im = im/np.max(im)
    
    fig = plt.figure(figsize=(5,5))
    plt.imshow(im,cmap='gray',aspect='equal',vmin=0, vmax=1)
    plt.imshow(msk)
    plt.suptitle(name)
    plt.tight_layout()
    fig.axes[0].set_axis_off()
    plt.show()

#%%
def mask_viewer(imvol,maskvol,name='Mask Display'):
    msksiz = np.r_[maskvol.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*maskvol.astype(float)
    
    imvol = imvol-np.min(imvol)
    imvol = imvol/np.max(imvol)
    
    fig = plt.figure(figsize=(5,5))
    fig.index = 0
    imobj = plt.imshow(imvol[fig.index,...],cmap='gray',aspect='equal',vmin=0, vmax=1)
    mskobj = plt.imshow(msk[fig.index,...])
    plt.tight_layout()
    plt.suptitle(name)
    ax = fig.axes[0]
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,fig.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    fig.imvol = imvol
    fig.maskvol = msk
    fig.imobj = imobj
    fig.mskobj = mskobj
    fig.txtobj = txtobj
    fig.canvas.mpl_connect('scroll_event',on_scroll_m0)
    
def on_scroll_m0(event):
    fig = event.canvas.figure
    if event.button == 'up':
        next_slice_m0(fig)
    elif event.button == 'down':
        previous_slice_m0(fig)
    fig.txtobj.set_text(fig.index+1)
    fig.canvas.draw()
    
def previous_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
    fig.index = np.max([np.min([fig.index-1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()

def next_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
    fig.index = np.max([np.min([fig.index+1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()

#%% Generalized Block Model
from keras.layers import Input, Cropping2D, Conv2D
from keras.layers import concatenate, BatchNormalization
from keras.layers import Conv2DTranspose, ZeroPadding2D
from keras.layers.advanced_activations import ELU
from keras.models import Model
init = 'he_normal'
def BlockModel(input_shape,filt_num=16,numBlocks=3):
    lay_input = Input(shape=(input_shape[1:]),name='input_layer')
        
     #calculate appropriate cropping
    mod = np.mod(input_shape[1:3],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[1:3]-padamt)
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    if minsize<4:
        raise ValueError('Too small of input for this many blocks. Use fewer blocks or larger input')
    
    crop = Cropping2D(cropping=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_input)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',kernel_initializer=init,name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',kernel_initializer=init,name='ConvAll_{}'.format(rr))(lay_merge)
#    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=init,name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-n 
    for rr in range(2,numBlocks+1):
        lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',kernel_initializer=init,name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',kernel_initializer=init,name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',kernel_initializer=init,name='ConvAll_{}'.format(rr))(lay_merge)
#        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(lay_conv_all)
        lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',kernel_initializer=init,strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
        
    # expanding block n
    dd=numBlocks
    lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',kernel_initializer=init,name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',kernel_initializer=init,name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',kernel_initializer=init,name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',kernel_initializer=init,name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',kernel_initializer=init,name='DeConvAll_{}'.format(dd))(lay_merge)
#    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(lay_deconv_all)
    lay_stride = Conv2DTranspose(filt_num*dd,(4,4),strides=(2,2),kernel_initializer=init,name='DeConvStride_{}'.format(dd))(lay_act)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
        
    # expanding blocks n-1
    expnums = list(range(1,numBlocks))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',kernel_initializer=init,name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',kernel_initializer=init,name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(filt_num*dd, (3, 3),padding='same',kernel_initializer=init,name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(filt_num*dd, (3, 3),padding='same',kernel_initializer=init,name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',kernel_initializer=init,name='DeConvAll_{}'.format(dd))(lay_merge)
#        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(lay_deconv_all)
        lay_stride = Conv2DTranspose(filt_num*dd,(4,4),strides=(2,2),kernel_initializer=init,name='DeConvStride_{}'.format(dd))(lay_act)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
                
    lay_pad = ZeroPadding2D(padding=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_act)
    lay_cleanup = Conv2D(filt_num,(3,3),padding='same',kernel_initializer=init,name='CleanUp_1')(lay_pad)
    lay_cleanup = Conv2D(filt_num,(3,3),padding='same',kernel_initializer=init,name='CleanUp_2')(lay_cleanup)
    # output
    lay_out = Conv2D(1,(1,1), activation='sigmoid',kernel_initializer=init,name='output_layer')(lay_cleanup)
    
    return Model(lay_input,lay_out)
