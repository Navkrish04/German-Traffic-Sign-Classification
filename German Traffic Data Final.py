
# coding: utf-8

# # German Traffic Sign Image Recognition 

# In[ ]:


import os
import time as time

import numpy as np
np.random.seed(40)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import time
from datetime import timedelta
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

import keras
import skimage.morphology as morp
from skimage.filters import rank
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout # new!
from keras.layers.normalization import BatchNormalization # new!
from keras import regularizers # new! 
from keras.optimizers import SGD
from keras.layers import Flatten, Conv2D, MaxPooling2D # new!
from keras.callbacks import ModelCheckpoint

import cv2

os.chdir("/Users/Navinkrishnakumar/Documents/Deep Learning")
os.getcwd()
 


# In[ ]:


import pickle
training_file = "/Users/Navinkrishnakumar/Documents/Deep Learning/Traffic_Sign_Dataset/train.p"
testing_file = "/Users/Navinkrishnakumar/Documents/Deep Learning/Traffic_Sign_Dataset/test.p"
validation_file = "/Users/Navinkrishnakumar/Documents/Deep Learning/Traffic_Sign_Dataset/valid.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_valid , y_valid = valid['features'], valid['labels']



# ## Understand the shape of the datasets
# 
# 

# In[ ]:


# Understand the data
print("Training Set:", len(X_train))
print("Test Set:", len(y_test))
print("Validation Set:", len(X_valid))
print("Image Dimensions:", np.shape(X_train[1]))
print("Number of classes:", len(np.unique(y_train)))
n_classes = len(np.unique(y_train))


# # Basic Descriptive analysis. 
# 
# ## Look for class bias 

# In[ ]:


# Basic Descriptive analysis


# Checking for class bias 

unique_elements, counts_elements = np.unique(y_train, return_counts = True)
print(np.asarray((unique_elements, counts_elements)))

pyplot.bar( np.arange( 43 ), counts_elements, align='center',color='green' )
pyplot.xlabel('Class')
pyplot.ylabel('No of Training data')
pyplot.xlim([-1, 43])

pyplot.show()

print(" ")
print("We can definitely see class bias issue as certain classes are under represented")

# View few images

import matplotlib.pyplot as plt
import random

get_ipython().run_line_magic('matplotlib', 'inline')
print(" ")
print(" ")
print("Traffic Sign Images")
fig, axs = plt.subplots(8,5, figsize=(10, 10))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(40):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
    
    



# ## Model Testing without any preprocessing - 
# 
# ## Establishing Baseline

# # Neural Network Architecture

# In[ ]:


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(32*32*3,)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(n_classes, activation='softmax'))


# In[ ]:


model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# In[ ]:


X_train_baseline = X_train.reshape(len(X_train), 32*32*3).astype('float32')
X_valid_baseline = X_valid.reshape(len(X_valid), 32*32*3).astype('float32')
y_train_baseline = keras.utils.to_categorical(y_train, n_classes)
y_valid_baseline = keras.utils.to_categorical(y_valid, n_classes)


# In[ ]:


model.fit(X_train_baseline, y_train_baseline, batch_size=128, epochs=100, verbose=1, validation_data=(X_valid_baseline, y_valid_baseline))


# # Data Preprocessing

# ## Data Augmentation
# ## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations

# In[ ]:


def data_augment(image):
    rows= image.shape[0]
    cols = image.shape[1]
    
    # rotation
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    
    # Translation
    M_trans = np.float32([[1,0,3],[0,1,6]])
    
    
    img = cv2.warpAffine(image,M_rot,(cols,rows))
    img = cv2.warpAffine(image,M_trans,(cols,rows))
    #img = cv2.warpAffine(image,M_aff,(cols,rows))
    
    # Bilateral filtering
    img = cv2.bilateralFilter(img,9,75,75)
    return img

"""
%matplotlib inline
print(" ")
print(" ")
print("Let's view few images to familiarize")
fig, axs = plt.subplots(8,5, figsize=(10, 10))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(40):
    index = np.random.randint(0, len(X_train))
    image = data_augment(X_train[index])
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
"""


# ## Randomly increase training set by data augmentation

# In[ ]:


# Data Augmentation with random data for X_train and y_train
X_aug = []
Y_aug = []
diff = 10000
def Augmentation(data1, data2,size):
    for x in range(size):
        index = np.random.randint(0,30000)
        img = data1[index]
        img_y = data2[index]
        trans_img = data_augment(img)
        X_aug.append(trans_img)
        Y_aug.append(img_y)
    return X_aug, Y_aug

# Augment the training sets
Augmentation(X_train,y_train,diff)

X_train_final = np.concatenate((X_train, X_aug), axis=0)
y_train_final = np.concatenate((y_train, Y_aug))




# ## Increase training set by augmentation and making all class size similar

# In[ ]:


classes = 43

X_train_final = X_train
y_train_final = y_train

for i in range(0,classes):
    
    class_records = np.where(y_train==i)[0].size
    max_records = 2010
    if class_records != max_records:
        ovr_sample = max_records - class_records
        samples = X_train[np.where(y_train==i)[0]]
        X_aug = []
        Y_aug = [i] * ovr_sample
        for x in range(ovr_sample):
            img = samples[x % class_records]
            trans_img = data_augment(img)
            X_aug.append(trans_img)
        X_train_final = np.concatenate((X_train_final, X_aug), axis=0)
        y_train_final = np.concatenate((y_train_final, Y_aug))    
  
    


# ## Check class bias after augmentation

# In[ ]:


unique_elements, counts_elements = np.unique(y_train_final, return_counts = True)
print(np.asarray((unique_elements, counts_elements)))

pyplot.bar( np.arange( 43 ), counts_elements, align='center',color='green' )
pyplot.xlabel('Class')
pyplot.ylabel('No of Training data')
pyplot.xlim([-1, 43])

pyplot.show()


# ## Shape of the datasets after data augmentation

# In[ ]:


print(len(X_train))
print(len(X_train_final))
print(len(y_train))
print(len(y_train_final))


# ## View images after data augmentation

# In[ ]:


import matplotlib.pyplot as plt
import random

get_ipython().run_line_magic('matplotlib', 'inline')
print(" ")
print(" ")
print("Let's view few images to familiarize")
fig, axs = plt.subplots(8,5, figsize=(10, 10))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(40):
    index = random.randint(0, len(X_train_final))
    image = X_train_final[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train_final[index])
    


# ## Model Testing after Augmentation

# In[ ]:


X_train_aug = X_train_final.reshape(len(X_train_final), 32*32*3).astype('float32')
X_valid_aug = X_valid.reshape(len(X_valid), 32*32*3).astype('float32')
y_train_aug = keras.utils.to_categorical(y_train_final, n_classes)
y_valid_aug = keras.utils.to_categorical(y_valid, n_classes)


# In[ ]:


model.fit(X_train_aug, y_train_aug, batch_size=64, epochs=100, verbose=1, validation_data=(X_valid_aug, y_valid_aug))


# # Gray Scaling

# In[ ]:


def gray_scale(image):
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#gray_images_data = list(map(gray_scale, X_train_final))

"""
%matplotlib inline
print(" ")
print(" ")
print("Let's view few images to familiarize")
fig, axs = plt.subplots(8,5, figsize=(10, 10))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(40):
    index = np.random.randint(0, len(gray_images_data))
    image = gray_images_data[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(Y[index])
  """



# In[ ]:


def local_histo_equalize(image):
    
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local

"""
local_histo = np.array(list(map(local_histo_equalize, gray_images_data)))
%matplotlib inline
print(" ")
print(" ")
print("Let's view few images to familiarize")
fig, axs = plt.subplots(8,5, figsize=(10, 10))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(40):
    index = np.random.randint(0, len(local_histo))
    image = local_histo[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(Y[index])
 """


# In[ ]:


def preprocess(data):
    gray_images = list(map(gray_scale, data))
    equalized_images = list(map(local_histo_equalize, gray_images))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = np.divide(img,255)
    normalized_images = normalized_images[..., None]
    return normalized_images


# ## Apply Grayscaling and local histogram equalization to the training and the validation data

# In[ ]:


X_train_preprocessed = preprocess(X_train_final)
X_valid_preprocessed = preprocess(X_valid)


# In[ ]:


X_train_preprocessed = X_train_preprocessed.reshape(len(X_train_preprocessed), 32*32*1).astype('float32')
X_valid_preprocessed = X_valid_preprocessed.reshape(len(X_valid_preprocessed), 32*32*1).astype('float32')


# In[ ]:


y_train_final = keras.utils.to_categorical(y_train_final, n_classes)
y_valid_final = keras.utils.to_categorical(y_valid, n_classes)


# ## Check the shape of the datasets after all preprocessing

# In[ ]:


print(X_train_preprocessed.shape)
print(X_valid_preprocessed.shape)
print(y_train_final.shape)
print(y_valid_final.shape)


# # Compile  and fit the model after preprocessing

# ## Neural network architecture after grayscaling

# In[ ]:


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(32*32*1,)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(n_classes, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ## Save model with Checkpoint

# In[ ]:


filepath="German_Traffic_DenseNetworkModel.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[ ]:


model.fit(X_train_preprocessed, y_train_final, batch_size=128, epochs=100, verbose=1,callbacks=callbacks_list,validation_data=(X_valid_preprocessed, y_valid_final))


# ## prepare test data for final results

# In[ ]:


## Prepare the Test data with all the preprocessing

X_test_preprocessed = preprocess(X_test)
X_test_preprocessed = X_test_preprocessed.reshape(len(X_test_preprocessed), 32*32*1).astype('float32')
y_test_final = keras.utils.to_categorical(y_test, n_classes)


# ## Load the best model from the validation data results

# In[ ]:


model.load_weights("German_Traffic_DenseNetworkModel.hdf5")


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


Pred = model.evaluate(X_test_preprocessed, y_test_final, verbose=0)
print("Dense fully connected network results on the test data")
print(" ")
print("%s- %.2f" % (model.metrics_names[0], Pred[0]))
print("%s- %.2f" % (model.metrics_names[1], Pred[1]))


# ##  Convolutional Networks

# In[ ]:


model_conv = Sequential()
## If You preprocessed with gray scaling and local histogram equivalization then input_shape = (32,32,1) else (32,32,3)
model_conv.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(32, 32, 1)))
model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(Dropout(0.25))
model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(Dropout(0.25))
model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(Dropout(0.25))
model_conv.add(Flatten())
model_conv.add(Dense(128, activation='relu'))
model_conv.add(Dropout(0.5))
model_conv.add(Dense(n_classes, activation='softmax'))


# In[ ]:


model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


filepath="German_Traffic_ConvNetworkModel.hdf5"
checkpoint_conv = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_conv = [checkpoint_conv]


# In[ ]:


model_conv.fit(X_train_preprocessed, y_train_final, batch_size=128, epochs=100, verbose=1,callbacks=callbacks_list_conv,validation_data=(X_valid_preprocessed, y_valid_final))


# In[ ]:


model_conv.load_weights("German_Traffic_ConvNetworkModel.hdf5")
model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


Pred_conv = model.evaluate(X_test_preprocessed, y_test_final, verbose=0)
print("Dense fully connected network results on the test data")
print(" ")
print("%s- %.2f" % (model.metrics_names[0], Pred_conv[0]))
print("%s- %.2f" % (model.metrics_names[1], Pred_conv[1]))

