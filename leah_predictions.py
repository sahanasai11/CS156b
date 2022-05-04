
# 2 model architecture using 10,000 training images and all testing
# both using alexnet
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# import np_utils # pip install np-utils

import cv2

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D


# In[ ]:


# -----------------------------------------------------------------------
# DATA UPLOAD
# -----------------------------------------------------------------------

# Total number of train and test points
width = 100
height = 100           
     
label_names = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
               "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia",
               "Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other",
               "Fracture","Support Devices"]

id_header = 'Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices'

# subset 10,000 training data
ROOT_PATH = '/central/groups/CS156b/teams/clnsh/subset_compressed_data10/subset_'

# full testing data
ROOT_PATH2 = '/central/groups/CS156b/teams/clnsh/compressed_data/'

print('uploading data...')
train_loaded = np.loadtxt(ROOT_PATH + 'train_matrices')
num_train = int(train_loaded.size / (width * height))
train = train_loaded.reshape(num_train, (width*height) // width, width)

print('train images loaded: ', len(train))

train_label = pd.read_table(ROOT_PATH + 'train_labels', delimiter=" ", 
              names=label_names)

print('train labels loaded: ', len(train_label))

test_loaded = np.loadtxt(ROOT_PATH2 + 'test_matrices')
num_test = int(test_loaded.size / (width * height))
test = test_loaded.reshape(num_test, (width*height) // width, width)

print('test images loaded: ', len(test))

#solution_loaded = np.loadtxt(ROOT_PATH2 + 'solution_matrices')
#num_solution = int(solution_loaded.size / (width * height))
#solution = solution_loaded.reshape(num_solution, (width*height) // width, width)

#print('solution images loaded: ', len(solution))

test_img_id = np.loadtxt(ROOT_PATH2 + 'test_img_ids')
#solution_img_id = np.loadtxt(ROOT_PATH2 + 'solution_img_ids')

print('test id loaded:', len(test_img_id))
#print('solution id loaded: ', len(solution_img_id))

# forming labels of training data
y_train = []
for index, row in train_label.iterrows():
    temp2 = [row['No Finding'], row['Enlarged Cardiomediastinum'], row['Cardiomegaly'],
            row['Lung Opacity'], row['Lung Lesion'], row['Edema'], row['Consolidation'],
            row['Pneumonia'], row['Atelectasis'], row['Pneumothorax'], row['Pleural Effusion'], 
            row['Pleural Other'], row['Fracture'], row['Support Devices']]
    i = 0 
    for val in temp2: 
        if val != val: # Handles NaN's
            temp2[i] = 0.0
        i += 1
    y_train.append(temp2)

print('y_train loaded: ', len(y_train), '\n\n')
X_train = train
y_train = np.array(y_train)


# In[ ]:


# making 1d-array of labels just for support devices and 2d-array of 13 other labels
y_train_support = []
y_train_other = []
i = 0
while i < len(y_train):
    y_train_support.append(y_train[i][13])
    y_train_other.append(y_train[i][:13])
    i += 1

y_train_support = np.array(y_train_support)
y_train_other = np.array(y_train_other)

print('number of label 1 support devices:', sum(y_train_support))
print('y_train_support: ', len(y_train_other))


# In[ ]:


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
# ALEX NET, SUPPORT DEVICES
# -----------------------------------------------------------------------
num_batch = 100
num_epoch = 15

model_support = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(100,100,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid')])

model_support.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.05),
    metrics=['mean_squared_error'])

history_support = model_support.fit(train_tensor_of_tensors, y_train_support, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_support = model_support.predict(test_tensor_of_tensors)
preds_support = preds_support.tolist()

print('support device predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# ALEX NET, 13 CLASSES
# -----------------------------------------------------------------------
num_classes = 13
num_batch = 64
num_epoch = 15

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(100,100,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes,activation='sigmoid')])

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])


history = model.fit(train_tensor_of_tensors, y_train_other, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds = model.predict(test_tensor_of_tensors)
preds = preds.tolist()

print('13 class predictions complete')


# In[ ]:


for i in range(len(tensor_test)):
    img_id = test_img_id[i]
    preds[i].append(preds_support[i][0])
    preds[i].insert(0, img_id)

np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("test_sub.csv", preds, delimiter=",", fmt='%f', header = id_header, comments = '')

for i in range(len(preds)): 
    row = preds[i]
    curr = row[2:]
    flag = False
    for val in curr: 
        if val >= 0.2: 
            flag = True
            break 
    if flag == True: 
        preds[i][1] = -1

np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("test_sub_nofinding.csv", preds, delimiter=",", fmt='%f', header = id_header, comments = '')

print('hi')


# In[ ]:


plt.plot(history_support.history['loss'])
plt.plot(history_support.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()

