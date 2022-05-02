
# coding: utf-8

# In[40]:


# turned normalization into a function, implemented class weights for support devices,
# divided into two models: one for support devices, and one for disease classification
import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image

# import np_utils # pip install np-utils

import cv2

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D


def load_data(Train_df,idx,
              batch_size):
    df = pd.read_csv(
                  Train_df, skiprows=idx*batch_size,
                  nrows=batch_size)
    x = df.iloc[:,1:]
         
    y = df.iloc[:,0]
    return (np.array(x), np_utils.to_categorical(y))


def batch_generator(Train_df,batch_size,
                    steps):
    idx=1
    while True: 
        yield load_data(Train_df,idx-1,batch_size)## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1

# -----------------------------------------------------------------------
# DATA UPLOAD
# -----------------------------------------------------------------------

# Total number of train and test points
MAX_TRAIN = 178158
MAX_TEST = 22596

SUBSET_TRAIN = 1000
SUBSET_TEST = 700

num_train = MAX_TRAIN
num_test = MAX_TEST

width = 100
height = 100           
     
label_names = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
               "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia",
               "Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other",
               "Fracture","Support Devices"]

id_header = 'Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices'

# For full data use:
ROOT_PATH = '/central/groups/CS156b/teams/clnsh/compressed_data/'

# For subset data use: 
# ROOT_PATH = '/central/groups/CS156b/teams/clnsh/subset_compressed_data/subset_'

print('uploading data...')
train_loaded = np.loadtxt(ROOT_PATH + 'train_matrices')
num_train = int(train_loaded.size / (width * height))
train = train_loaded.reshape(num_train, (width*height) // width, width)

print('train images loaded: ', len(train))

train_label = pd.read_table(ROOT_PATH + 'train_labels', delimiter=" ", 
              names=label_names)

print('train labels loaded: ', len(train_label))

test_loaded = np.loadtxt(ROOT_PATH + 'test_matrices')
num_test = int(test_loaded.size / (width * height))
test = test_loaded.reshape(num_test, (width*height) // width, width)

print('test images loaded: ', len(test))

solution_loaded = np.loadtxt(ROOT_PATH + 'solution_matrices')
num_solution = int(solution_loaded.size / (width * height))
solution = solution_loaded.reshape(num_solution, (width*height) // width, width)

print('solution images loaded: ', len(solution))

test_img_id = np.loadtxt(ROOT_PATH + 'test_img_ids')
solution_img_id = np.loadtxt(ROOT_PATH + 'solution_img_ids')

print('test, solution ids loaded: ', len(test_img_id), ',', len(solution_img_id))

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


# In[41]:


# making 1d-array of labels just for support devices and 2d-array of 13 other labels
y_train_support = []
y_train_other = []
i = 0
while i < len(y_train):
    y_train_support.append(max(0, y_train[i][13]))
    y_train_other.append(y_train[i][:13])
    i += 1

y_train_support = np.array(y_train_support)
y_train_other = np.array(y_train_other)

print('y_train_support, y_train_other loaded:', sum(y_train_support), ',', len(y_train_other))

# forming class weights for support devices (0: has no support device, 1: has support device)
# class_weight_support = {0: 1/(len(y_train_support) - sum(y_train_support)), 1: 1/sum(y_train_support)}
# print(class_weight_support)
# print('class weights formed')


# In[42]:


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

for i in range(0, len(solution)):
    tensor_solution.append(tf.convert_to_tensor(solution[i], dtype=float))
print('solution tensors created')

# setting training and testing data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
train_tensor_of_tensors = tf.convert_to_tensor(tensor_train)
print('train tensor of tensor created:', train_tensor_of_tensors.shape)

test_tensor_of_tensors = tf.convert_to_tensor(tensor_test)
print('test tensors of tensors created:', test_tensor_of_tensors.shape)


# In[43]:


# In[ ]:


# -----------------------------------------------------------------------
# MSE MODEL, 13 CLASSES
# -----------------------------------------------------------------------

num_classes = 13
# FOR WHOLE DATA:
num_batch = 2000

# FOR SUBSET DATA:
#num_batch = 200
num_epoch = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=[width, height, 1]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='relu'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.compile(loss= tf.keras.losses.MeanSquaredError(), # keras.losses.categorical_crossentropy, mse loss
#               optimizer= keras.optimizers.Adadelta(),
#               # keras.optimizers.Adadelta(), # keras.optimizers.Adam()
#               metrics=['mean_squared_error']) # MSE accuracy

model.fit(train_tensor_of_tensors, y_train_other, epochs=num_epoch, verbose=1, batch_size=num_batch)


preds_other = model.predict(test_tensor_of_tensors)
preds_other = preds_other.tolist() 

print('predictions complete')

# -----------------------------------------------------------------------
# MODEL FOR SUPPORT DEVICE, BINARY
# -----------------------------------------------------------------------
num_classes_support = 1
num_batch_support = 1000
num_epoch_support = 7

model_binary = Sequential()
model_binary.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=[width, height, 1]))
model_binary.add(Conv2D(64, (3, 3), activation='relu'))
model_binary.add(MaxPooling2D(pool_size=(2, 2)))
model_binary.add(Flatten())
model_binary.add(Dense(128, activation='relu'))
model_binary.add(Dense(num_classes_support, activation='sigmoid'))

model_binary.compile(loss= tf.keras.losses.binary_crossentropy,
              optimizer= keras.optimizers.Adam(),
              # keras.optimizers.Adadelta(),
              metrics=['accuracy']) # MSE accuracy

model_binary.fit(train_tensor_of_tensors, y_train_support, epochs=num_epoch_support, 
                 verbose=1, batch_size=num_batch_support) #class_weight = class_weight_support
print('model fit complete')


preds_support = model_binary.predict(test_tensor_of_tensors)
preds_support = preds_support.tolist()

# In[ ]:


# scaling the predictions to be between -1 and 1

def prediction_normalization(preds_array):
    scaled_preds = []
    for i in range(len(preds_array)):
        curr = np.array(preds_array[i])
        factor = min(curr)
        newList = [(x - factor) for x in curr]
        max_curr = max(newList)
        myInt = max_curr / 2
        newList2 = [((x / myInt)-1) for x in newList]
        scaled_preds.append(newList2)
    return scaled_preds

scaled_preds_other = prediction_normalization(preds_other)
print('scaled predictions complete')

print(scaled_preds_other[0])

for i in range(len(tensor_test)):
    img_id = test_img_id[i]
    scaled_preds_other[i].append(preds_support[i][0])
    scaled_preds_other[i].insert(0, img_id)

print(preds_support[0])
print(scaled_preds_other[0])
np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("test_sub.csv", scaled_preds_other, delimiter=",", fmt='%f', header = id_header, comments = '')
print('hi')
