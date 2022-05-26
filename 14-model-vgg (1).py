
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import copy

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPool2D


# In[ ]:


# -----------------------------------------------------------------------
# DATA UPLOAD
# -----------------------------------------------------------------------  
label_names = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
               "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia",
               "Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other",
               "Fracture","Support Devices"]

id_header = 'Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices'
 
# -----------------------------------------------------------------------
# FOR ROOT PATHS: EITHER COMMENT OUT BLOCK USING 100 x 100 or 224 x 224
# -----------------------------------------------------------------------
width = 100
height = 100   

# subset 20,000 training data ( 100 x 100 )
TRAIN_PATH = '/central/groups/CS156b/teams/clnsh/subset_compressed_data20/subset_'
TRAIN_LABELS_PATH = '/central/groups/CS156b/teams/shncl/subset_compressed_data/subset_'

# full testing data ( 100 x 100 )
TEST_PATH = '/central/groups/CS156b/teams/clnsh/compressed_data/'
# -----------------------------------------------------------------------
# width = 224
# height = 224

# # subset 10,000 training data ( 224 x 224 )
# TRAIN_PATH ='/central/groups/CS156b/teams/clnsh/subset_compressed_224_train/'

# # full testing data ( 224 x 224 )
# TEST_PATH = '/central/groups/CS156b/teams/shncl2/'
# -----------------------------------------------------------------------

# Training data:
print('uploading data...')
train_loaded = np.loadtxt(TRAIN_PATH + 'train_matrices')
num_train = int(train_loaded.size / (width * height))
train = train_loaded.reshape(num_train, (width*height) // width, width)
train = np.repeat(train[..., np.newaxis], 3, -1)

print('train images loaded: ', len(train))

#used to be train_path + train_labels
train_label = pd.read_table(TRAIN_LABELS_PATH + 'train_labels_nan', delimiter=" ", 
              names=label_names)

print('train labels loaded: ', len(train_label))

# Testing data:
test_loaded = np.loadtxt(TEST_PATH + 'test_matrices') # NOTE: 224 is added
num_test = int(test_loaded.size / (width * height))
test = test_loaded.reshape(num_test, (width*height) // width, width)
test = np.repeat(test[..., np.newaxis], 3, -1)

print('test images loaded: ', len(test))

test_img_id = np.loadtxt(TEST_PATH + 'test_img_ids')

print('test id loaded:', len(test_img_id))

# Solution data:
#solution_loaded = np.loadtxt(TEST_PATH + 'solution_matrices')
#num_solution = int(solution_loaded.size / (width * height))
#solution = solution_loaded.reshape(num_solution, (width*height) // width, width)
#print('solution images loaded: ', len(solution))

#solution_img_id = np.loadtxt(TEST_PATH + 'solution_img_ids')
#print('solution id loaded: ', len(solution_img_id))


# -----------------------------------------------------------------------
# FORMING LABELS
# -----------------------------------------------------------------------

# forming labels of training data
y_train = []
for index, row in train_label.iterrows():
    temp2 = [row['No Finding'], row['Enlarged Cardiomediastinum'], row['Cardiomegaly'],
            row['Lung Opacity'], row['Lung Lesion'], row['Edema'], row['Consolidation'],
            row['Pneumonia'], row['Atelectasis'], row['Pneumothorax'], row['Pleural Effusion'], 
            row['Pleural Other'], row['Fracture'], row['Support Devices']]
    i = 0 
    #for val in temp2: 
        #if val != val: # Handles NaN's
            #temp2[i] = 0.0
        #i += 1
    y_train.append(temp2)

print('y_train loaded: ', len(y_train), '\n\n')
X_train = train
y_train = np.array(y_train)

# scaling the labels to be 0, 0.5, 1
y_train_scaled = copy.deepcopy(y_train)
i = 0
while i < len(y_train_scaled):
    y_train_scaled[i] = (y_train_scaled[i] + 1) / 2
    i += 1
    
y_train_scaled = np.array(y_train_scaled)

# In[95]:


# making 14 separate sub-arrays splitting the training lables
y_train_no_finding = []
y_train_enlarged_cardio = []
y_train_cardio = []
y_train_lung_opacity = []
y_train_lung_lesion = []
y_train_edema = []
y_train_consolidation = []
y_train_pneumonia = []
y_train_atelectasis = []
y_train_pneumothorax = []
y_train_pleural_effusion = []
y_train_pleural_other = []
y_train_fracture = []
y_train_support_devices = []


i = 0
while i < len(y_train_scaled):
    
    y_train_no_finding.append(y_train_scaled[i][0])
    y_train_enlarged_cardio.append(y_train_scaled[i][1])
    y_train_cardio.append(y_train_scaled[i][2])
    y_train_lung_opacity.append(y_train_scaled[i][3])
    y_train_lung_lesion.append(y_train_scaled[i][4])
    y_train_edema.append(y_train_scaled[i][5])
    y_train_consolidation.append(y_train_scaled[i][6])
    y_train_pneumonia.append(y_train_scaled[i][7])
    y_train_atelectasis.append(y_train_scaled[i][8])
    y_train_pneumothorax.append(y_train_scaled[i][9])
    y_train_pleural_effusion.append(y_train_scaled[i][10])
    y_train_pleural_other.append(y_train_scaled[i][11])
    y_train_fracture.append(y_train_scaled[i][12])
    y_train_support_devices.append(y_train_scaled[i][13])
   
    i += 1

y_train_no_finding = np.array(y_train_no_finding)
y_train_enlarged_cardio = np.array(y_train_enlarged_cardio)
y_train_cardio = np.array(y_train_cardio)
y_train_lung_opacity = np.array(y_train_lung_opacity)
y_train_lung_lesion = np.array(y_train_lung_lesion)
y_train_edema = np.array(y_train_edema)
y_train_consolidation = np.array(y_train_consolidation)
y_train_pneumonia = np.array(y_train_pneumonia)
y_train_atelectasis = np.array(y_train_atelectasis)
y_train_pneumothorax = np.array(y_train_pneumothorax)
y_train_pleural_effusion = np.array(y_train_pleural_effusion)
y_train_pleural_other = np.array(y_train_pleural_other)
y_train_fracture = np.array(y_train_fracture)
y_train_support_devices = np.array(y_train_support_devices)

print('Subarrays formed')

# In[97]:


# Forming lists of tensored images
print('initializing tensors...')
tensor_train = []
tensor_test = []
tensor_solution = []

for i in range(0, len(train)):
    tensor_train.append(tf.convert_to_tensor(train[i], dtype=float))
print('train tensors created')

for i in range(0, len(test)):
    tensor_test.append(tf.convert_to_tensor(test[i], dtype=float))
print('test tensors created')

# for i in range(0, len(solution)):
#     tensor_solution.append(tf.convert_to_tensor(solution[i], dtype=float))
# print('solution tensors created')

# setting training and testing data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
train_tensor_of_tensors = tf.convert_to_tensor(tensor_train)
print('train tensor of tensor created:', train_tensor_of_tensors.shape)

test_tensor_of_tensors = tf.convert_to_tensor(tensor_test)
print('test tensors of tensors created:', test_tensor_of_tensors.shape)


# In[ ]:


# -----------------------------------------------------------------------
# VGG: No Finding
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_no_finding = Model (inputs=input, outputs =output)

model_no_finding.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_model_no_finding = model_no_finding.fit(train_tensor_of_tensors, y_train_no_finding, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_no_finding = model_no_finding.predict(test_tensor_of_tensors)
preds_no_finding= preds_no_finding.tolist()
print('no finding predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Enlarged Cardiomediastinum
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_enlarged_cardio = Model (inputs=input, outputs =output)

model_enlarged_cardio.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_enlarged_cardio = model_enlarged_cardio.fit(train_tensor_of_tensors, y_train_enlarged_cardio, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_enlarged_cardio = model_enlarged_cardio.predict(test_tensor_of_tensors)
preds_enlarged_cardio = preds_enlarged_cardio.tolist()
print('enlarged_cardio predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Cardio
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_cardio = Model (inputs=input, outputs =output)

model_cardio.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_cardio = model_cardio.fit(train_tensor_of_tensors, y_train_cardio, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_cardio = model_cardio.predict(test_tensor_of_tensors)
preds_cardio = preds_cardio.tolist()
print('cardio predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Lung Opacity
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_lung_opacity = Model (inputs=input, outputs =output)

model_lung_opacity.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_lung_opacity = model_lung_opacity.fit(train_tensor_of_tensors, y_train_lung_opacity, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_lung_opacity = model_lung_opacity.predict(test_tensor_of_tensors)
preds_lung_opacity = preds_lung_opacity.tolist()
print('Lung opacity predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Lung Lesion
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_lung_lesion = Model (inputs=input, outputs =output)

model_lung_lesion.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_lung_lesion = model_lung_lesion.fit(train_tensor_of_tensors, y_train_lung_lesion, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_lung_lesion = model_lung_lesion.predict(test_tensor_of_tensors)
preds_lung_lesion = preds_lung_lesion.tolist()
print('lung lesion predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Edema
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_edema = Model (inputs=input, outputs =output)

model_edema.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_edema = model_edema.fit(train_tensor_of_tensors, y_train_edema, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_edema = model_edema.predict(test_tensor_of_tensors)
preds_edema = preds_edema.tolist()
print('edema predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Consolidation
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_consolidation = Model (inputs=input, outputs =output)

model_consolidation.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_consolidation = model_consolidation.fit(train_tensor_of_tensors, y_train_consolidation, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_consolidation = model_consolidation.predict(test_tensor_of_tensors)
preds_consolidation = preds_consolidation.tolist()
print('consolidation predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: pneumonia
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_pneumonia = Model (inputs=input, outputs =output)

model_pneumonia.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_pneumonia = model_pneumonia.fit(train_tensor_of_tensors, y_train_pneumonia, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_pneumonia = model_pneumonia.predict(test_tensor_of_tensors)
preds_pneumonia = preds_pneumonia.tolist()
print('pneumonia predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: atelectasis
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_atelectasis = Model (inputs=input, outputs =output)

model_atelectasis.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_atelectasis = model_atelectasis.fit(train_tensor_of_tensors, y_train_atelectasis, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_atelectasis = model_atelectasis.predict(test_tensor_of_tensors)
preds_atelectasis = preds_atelectasis.tolist()
print('atelectasis predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Pneumothorax
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_pneumothorax = Model (inputs=input, outputs =output)

model_pneumothorax.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_pneumothorax = model_pneumothorax.fit(train_tensor_of_tensors, y_train_pneumothorax, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_pneumothorax = model_pneumothorax.predict(test_tensor_of_tensors)
preds_pneumothorax = preds_pneumothorax.tolist()
print('pneumothorax predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: pleural_effusion
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_pleural_effusion = Model (inputs=input, outputs =output)

model_pleural_effusion.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_pleural_effusion = model_pleural_effusion.fit(train_tensor_of_tensors, y_train_pleural_effusion, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_pleural_effusion = model_pleural_effusion.predict(test_tensor_of_tensors)
preds_pleural_effusion = preds_pleural_effusion.tolist()
print('pleural_effusion predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: pleural_other
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_pleural_other = Model (inputs=input, outputs =output)

model_pleural_other.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_pleural_other = model_pleural_other.fit(train_tensor_of_tensors, y_train_pleural_other, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_pleural_other = model_pleural_other.predict(test_tensor_of_tensors)
preds_pleural_other = preds_pleural_other.tolist()
print('pleural_other predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Fracture
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_fracture = Model (inputs=input, outputs =output)

model_fracture.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_fracture = model_fracture.fit(train_tensor_of_tensors, y_train_fracture, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_fracture = model_fracture.predict(test_tensor_of_tensors)
preds_fracture = preds_fracture.tolist()
print('fracture predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# VGG: Support Devices
# -----------------------------------------------------------------------

num_epoch = 15
num_batch = 128
num_classes = 1

input = Input(shape =(100,100,3))
# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = num_classes, activation ='sigmoid')(x)

model_support_devices = Model (inputs=input, outputs =output)

model_support_devices.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])
    
history_support_devices = model_support_devices.fit(train_tensor_of_tensors, y_train_support_devices, validation_split = 0.2, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_support_devices = model_support_devices.predict(test_tensor_of_tensors)
preds_support_devices = preds_support_devices.tolist()
print('support_devices')


# In[ ]:

combined = [[] for _ in range(15)]

img_id = test_img_id[i]

combined[0] = copy.deepcopy(test_img_id)
combined[1] = copy.deepcopy(preds_no_finding)
combined[2] = copy.deepcopy(preds_enlarged_cardio)
combined[3] = copy.deepcopy(preds_cardio)
combined[4] = copy.deepcopy(preds_lung_opacity)
combined[5] = copy.deepcopy(preds_lung_lesion)
combined[6] = copy.deepcopy(preds_edema)
combined[7] = copy.deepcopy(preds_consolidation)
combined[8] = copy.deepcopy(preds_pneumonia)
combined[9] = copy.deepcopy(preds_atelectasis)
combined[10] = copy.deepcopy(preds_pneumothorax)
combined[11] = copy.deepcopy(preds_pleural_effusion)
combined[12] = copy.deepcopy(preds_pleural_other)
combined[13] = copy.deepcopy(preds_fracture)
combined[14] = copy.deepcopy(preds_support_devices)
    
combined = np.transpose(combined)
    
print(combined[0], '\n')
np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("noscaling.csv", combined, delimiter=",", fmt='%f', header = id_header, comments = '')

combined_scaled = copy.deepcopy(combined)

# scaling the predictions to be between -1 and 1
for i in range(len(combined_scaled)):
    row = combined_scaled[i][1:]
    row = [((elem * 2) - 1) for elem in row]
    for j in range(len(row)):
        #round no findings to -1 
        if j == 0 and row[j] <= -0.5:
            row[j] = -1
        elif row[j] <= -0.8:
            row[j] = row[j] / 2
    combined_scaled[i][1:] = row
    
print(combined_scaled[0])

np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("scaled.csv", combined_scaled, delimiter=",", fmt='%f', header = id_header, comments = '')


# In[ ]:


# -----------------------------------------------------------------------
# LOSS PLOTTING
# -----------------------------------------------------------------------

plt.plot(history_pneumothorax.history['loss'])
plt.plot(history_pneumothorax.history['val_loss'])
plt.title('pneumothorax model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('pneumothorax_model_loss.png')
plt.show()


# In[ ]:


plt.plot(history_support.history['loss'])
plt.plot(history_support.history['val_loss'])
plt.title('support devices model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('support_model_loss.png')
plt.show()


# In[ ]:


plt.plot(history_cardio.history['loss'])
plt.plot(history_cardio.history['val_loss'])
plt.title('cardio model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cardio_model_loss.png')
plt.show()


# In[ ]:


plt.plot(history_p.history['loss'])
plt.plot(history_p.history['val_loss'])
plt.title('p model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('p_model_loss.png')
plt.show()


# In[ ]:


plt.plot(history_frac.history['loss'])
plt.plot(history_frac.history['val_loss'])
plt.title('fracture model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('fracture_model_loss.png')
plt.show()


# In[ ]:


plt.plot(history_nofinding.history['loss'])
plt.plot(history_nofinding.history['val_loss'])
plt.title('no finding model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('no_finding_model_loss.png')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('other classification model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('other_model_loss.png')
plt.show()

