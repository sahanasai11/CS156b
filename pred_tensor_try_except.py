
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install opencv-python')
get_ipython().system('pip3 install tensorflow')


# In[1]:


import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image

import cv2

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D


# In[2]:


# Load data

train = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')

test_ids = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')

solution_ids = pd.read_csv('/groups/CS156b/data/student_labels/solution_ids.csv')


# In[3]:


# Use subset of the data for models

# Total number of train and test points
MAX_TRAIN = 178158
MAX_TEST = 22596

num_train = MAX_TRAIN
num_test = MAX_TEST

train_sub = train.sample(n=num_train)
test_sub = test_ids.sample(n=num_test)
solution_ids_sub = solution_ids.sample(n=num_test)


# In[4]:


tensor_train = []
tensor_test = []
tensor_solution = []
width = 100
height = 100

y_train = []


# In[5]:


# Convert all training images into numpy arrays

init_path = '/groups/CS156b/data/'
c=0

for i, row in train_sub.iterrows():
    c+=1
    if c % 1000 == 0:
        print(c/num_train)
    try:
        temp_img = asarray(Image.open(init_path + row['Path']))
        temp_img = cv2.resize(temp_img, (width, height))
        temp_img = temp_img/255
        tensor_train.append(tf.convert_to_tensor(temp_img, dtype=float))
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
    except FileNotFoundError:
        print("Image doesn't exist")

print("Train Images Done")


# In[6]:


init_path = '/groups/CS156b/data/'

test_img_ids = []

for i, row in test_sub.iterrows():
    try:
        temp_img = asarray(Image.open(init_path + row['Path']))
        temp_img = cv2.resize(temp_img, (width, height))
        temp_img = temp_img/255
        tensor_test.append(tf.convert_to_tensor(temp_img, dtype=float))
        test_img_ids.append(row['Id'])
    except FileNotFoundError:
        print("Image doesn't exist")

print("Test Images Done")


# In[7]:


init_path = '/groups/CS156b/data/'

solution_img_ids = []

for i, row in solution_ids_sub.iterrows():
    try:
        temp_img = asarray(Image.open(init_path + row['Path']))
        temp_img = cv2.resize(temp_img, (width, height))
        temp_img = temp_img/255
        tensor_solution.append(tf.convert_to_tensor(temp_img, dtype=float))
        solution_img_ids.append(row['Id'])
    except FileNotFoundError:
        print("Image doesn't exist")

print("Solution Images Done")


# In[8]:


# setting training data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
train_tensor_of_tensors = tf.convert_to_tensor(tensor_train)
y_train = np.array(y_train)
print(train_tensor_of_tensors.shape)


# In[9]:


num_classes = 14

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=[width, height, 1]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_tensor_of_tensors, y_train, epochs=10, verbose=1, batch_size=64)


# In[10]:


# setting testing data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
test_tensor_of_tensors = tf.convert_to_tensor(tensor_test)

preds = model.predict(test_tensor_of_tensors)

print(len(tensor_test))

preds = preds.tolist()

result = []


for i in range(len(tensor_test)):
    img_id = test_img_ids[i]
    preds[i].insert(0, img_id)


# In[11]:


np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("test_sub.csv", preds, delimiter=",", fmt='%f')

