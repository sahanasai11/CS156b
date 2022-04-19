
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install opencv-python')
get_ipython().system('pip3 install tensorflow')


# In[2]:


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


# In[3]:


# Load data

train = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')

test_ids = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')

solution_ids = pd.read_csv('/groups/CS156b/data/student_labels/solution_ids.csv')


# In[4]:


# Use subset of the data for models

# Total number of train and test points
MAX_TRAIN = 178158
MAX_TEST = 22596

num_train = 1000
num_test = MAX_TEST

train_sub = train.sample(n=num_train)
test_sub = test_ids.sample(n=num_test)
solution_ids_sub = solution_ids.sample(n=num_test)


# In[5]:


# Convert all training images into numpy arrays

train_imgs = []
init_path = '/groups/CS156b/data/'

for i, row in train_sub.iterrows():
    temp_img = Image.open(init_path + row['Path'])
    train_imgs.append(asarray(temp_img))
    
    
train_imgs = np.array(train_imgs)


# In[ ]:


test_imgs = []
init_path = '/groups/CS156b/data/'

test_img_ids = []

for i, row in test_sub.iterrows():
    temp_img = Image.open(init_path + row['Path'])
    test_imgs.append(asarray(temp_img))
    test_img_ids.append(row['Id'])
    
test_imgs = np.array(test_imgs)


# In[ ]:


solution_imgs = []
init_path = '/groups/CS156b/data/'

solution_img_ids = []

for i, row in solution_ids_sub.iterrows():
    temp_img = Image.open(init_path + row['Path'])
    solution_imgs.append(asarray(temp_img))
    solution_img_ids.append(row['Id'])
    
solution_imgs = np.array(solution_imgs)


# In[ ]:


print(train_imgs[0].shape)


# In[ ]:


img_large = Image.fromarray(train_imgs[0], 'L')


# In[ ]:


img_large


# In[ ]:


width = 100
height = 100

curr = cv2.resize(train_imgs[0], (width, height))

print(curr.shape)
print(curr)

img = Image.fromarray(curr, 'L')


# In[ ]:


img


# In[ ]:


tensor_train = []
tensor_test = []
tensor_solution = []


# In[ ]:


for i in range(0, len(train_imgs)):
    train_imgs[i] = cv2.resize(train_imgs[i], (width, height))
    train_imgs[i] = train_imgs[i]/255
    tensor_train.append(tf.convert_to_tensor(train_imgs[i], dtype=float))
    
for i in range(0, len(test_imgs)):
    test_imgs[i] = cv2.resize(test_imgs[i], (width, height))
    test_imgs[i] = test_imgs[i]/255
    tensor_test.append(tf.convert_to_tensor(test_imgs[i], dtype=float))

    
for i in range(0, len(solution_imgs)):
    solution_imgs[i] = cv2.resize(solution_imgs[i], (width, height))
    solution_imgs[i] = solution_imgs[i]/255
    tensor_solution.append(tf.convert_to_tensor(solution_imgs[i], dtype=float))


# In[ ]:


y_train = []
for index, row in train_sub.iterrows():
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


# In[ ]:


X_train = train_imgs
y_train = np.array(y_train)

print(type(X_train[0][0]))


# In[ ]:


# setting training data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
train_tensor_of_tensors = tf.convert_to_tensor(tensor_train)
print(train_tensor_of_tensors.shape)


# In[ ]:


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

model.fit(train_tensor_of_tensors, y_train, epochs=10, verbose=0)


# In[ ]:


# setting testing data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
test_tensor_of_tensors = tf.convert_to_tensor(tensor_test)

preds = model.predict(test_tensor_of_tensors)

preds = preds.tolist()

result = []


for i in range(len(tensor_test)):
    img_id = test_img_ids[i]
    preds[i].insert(0, img_id)


# In[ ]:


np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("test_sub.csv", preds, delimiter=",", fmt='%f')

