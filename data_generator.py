#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from keras.utils import Sequence
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import numpy as np
import cv2
from keras.utils import Sequence
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import keras
import nibabel as nib
from PIL import Image
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum
from keras.layers import Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
from keras.layers import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D
from keras.layers import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam



def standardize(image):

    standardized_image = np.zeros(image.shape)

    #
   
        # iterate over the `z` dimension
    for z in range(image.shape[2]):
        # get a slice of the image
        # at channel c and z-th dimension `z`
        image_slice = image[:,:,z]

        # subtract the mean from image_slice
        centered = image_slice - np.mean(image_slice)
       
        # divide by the standard deviation (only if it is different from zero)
        if(np.std(centered)!=0):
            centered = centered/np.std(centered)

        # update  the slice of standardized image
        # with the scaled centered and scaled image
        standardized_image[:, :, z] = centered

    ### END CODE HERE ###

    return standardized_image


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self,
                 to_fit=True, batch_size=1, dim=(240, 240),
                 n_channels=4, n_classes=4, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        #self.list_IDs = list_IDs
        #self.labels = labels
        #self.image_path = image_path
        #self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x_to = []
        self.y_to = []
        #self.on_epoch_end()
       
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return 180//self.batch_size
   
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
       
        path = 'BRATS2017/Brats17TrainingData/HGG'
        all_images = os.listdir(path)
        #print(len(all_images),all_images[0])
        all_images = [img for img in all_images if not img.endswith('.DS_Store')]
        all_images.sort()
        data = np.zeros((240,240,155,4))
        
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            
            if i >= len(all_images):
                break
               
            image_filename = all_images[i]
            folder_path = os.path.join(path, image_filename)
            modalities = os.listdir(folder_path)
            modalities = [mod for mod in modalities if not mod.endswith('.DS_Store')]
            modalities.sort()
            w = 0
        
            for j in range(len(modalities) - 1):
                image_path = folder_path + '/' + modalities[j]
                print(image_path)
                if not(image_path.find('seg.nii') == -1):
                    img = nib.load(image_path)
                    image_data2 = img.get_fdata()
                    image_data2 = np.asarray(image_data2)
                    print("Entered ground truth")
                else:
                    if w < 4:  # Ensure w doesn't exceed the 4th dimension limit
                        img = nib.load(image_path)
                        image_data = img.get_fdata()
                        image_data = np.asarray(image_data)
                        image_data = standardize(image_data)
                        data[:, :, :, w] = image_data
                        print("Entered modality")
                        w += 1

            print(data.shape)
            print(image_data2.shape)  

            for slice_no in range(0, 155):
                X = data[:, :, slice_no, :]
                Y = image_data2[:, :, slice_no]

                if(X.any() != 0 and Y.any() != 0 and len(np.unique(Y)) == 4):
                    self.x_to.append(X)
                    self.y_to.append(Y.reshape(240, 240, 1))
                    if len(self.x_to) >= 60:
                        break

        x_to = np.asarray(self.x_to)
        y_to = np.asarray(self.y_to)
        x_to, y_to = shuffle(x_to, y_to)
   
        print(x_to.shape)
        print(y_to.shape)

        return x_to, y_to

