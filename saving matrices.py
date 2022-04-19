
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

num_train = MAX_TRAIN
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


# In[6]:


test_imgs = []
init_path = '/groups/CS156b/data/'

test_img_ids = []

for i, row in test_sub.iterrows():
    temp_img = Image.open(init_path + row['Path'])
    test_imgs.append(asarray(temp_img))
    test_img_ids.append(row['Id'])
    
test_imgs = np.array(test_imgs)


# In[7]:


solution_imgs = []
init_path = '/groups/CS156b/data/'

solution_img_ids = []

for i, row in solution_ids_sub.iterrows():
    temp_img = Image.open(init_path + row['Path'])
    solution_imgs.append(asarray(temp_img))
    solution_img_ids.append(row['Id'])
    
solution_imgs = np.array(solution_imgs)


# In[8]:


print(train_imgs[0].shape)


# In[9]:


img_large = Image.fromarray(train_imgs[0], 'L')


# In[10]:


img_large


# In[12]:


img


# In[50]:


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


# In[11]:


width = 100
height = 100

curr = cv2.resize(train_imgs[0], (width, height))

print(curr.shape)
print(curr)

img = Image.fromarray(curr, 'L')


# In[51]:


for i in range(0, len(train_imgs)):
    train_imgs[i] = cv2.resize(train_imgs[i], (width, height))
    train_imgs[i] = train_imgs[i]/255
    
for i in range(0, len(test_imgs)):
    test_imgs[i] = cv2.resize(test_imgs[i], (width, height))
    test_imgs[i] = test_imgs[i]/255

    
for i in range(0, len(solution_imgs)):
    solution_imgs[i] = cv2.resize(solution_imgs[i], (width, height))
    solution_imgs[i] = solution_imgs[i]/255


# In[67]:


train_df = pd.DataFrame(train_imgs, columns=['Pixel Matrix'])

train_df.to_csv('train_pixels')

