
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image

import cv2

# Load data

train = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')

test = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')

solution = pd.read_csv('/groups/CS156b/data/student_labels/solution_ids.csv')

train.head()


# Use subset of the data for models

# Total number of train and test points
MAX_TRAIN = 178158
MAX_TEST = 22596

num_train = 300
num_test = 100
# train_sub = train.sample(n=num_train)
# test_sub = test_ids.sample(n=num_test)
# solution_ids_sub = solution_ids.sample(n=num_test)

train_imgs = []
test_imgs = []
solution_imgs = []
width = 100
height = 100


# Convert all training images into numpy arrays

init_path = '/groups/CS156b/data/'

j = 0
for index, row in train.iterrows():
    if j >= num_train:
        break
    j += 1
    try:
        temp_img = asarray(Image.open(init_path + row['Path']))
        temp_img = cv2.resize(temp_img, (width, height))
        temp_img = temp_img/255
        train_imgs.append(temp_img)
        
    except FileNotFoundError:
        print("Image doesn't exist")

train_imgs = np.array(train_imgs)
reshaped_train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)

print("Train Images Done")

# NOTE: train_imgs is reshaped to be a 2d list instead of a 3d list


train_labels = []
j = 0
for index, row in train.iterrows():
    if j >= num_train:
        break
    j += 1
    temp2 = [row['No Finding'], row['Enlarged Cardiomediastinum'], row['Cardiomegaly'],
            row['Lung Opacity'], row['Lung Lesion'], row['Edema'], row['Consolidation'],
            row['Pneumonia'], row['Atelectasis'], row['Pneumothorax'], row['Pleural Effusion'], 
            row['Pleural Other'], row['Fracture'], row['Support Devices']]
    i = 0 
    for val in temp2: 
        if val != val: # Handles NaN's
            temp2[i] = 0.0
        i += 1
    train_labels.append(temp2)
    
print("Train labels done")


init_path = '/groups/CS156b/data/'

test_img_ids = []

j = 0
for i, row in test.iterrows():
    if j >= num_test:
        break
    j += 1
    try:
        temp_img = asarray(Image.open(init_path + row['Path']))
        temp_img = cv2.resize(temp_img, (width, height))
        temp_img = temp_img/255
        test_imgs.append(temp_img)
        test_img_ids.append(row['Id'])
    except FileNotFoundError:
        print("Image doesn't exist")

test_imgs = np.array(test_imgs)
reshaped_test_imgs = test_imgs.reshape(test_imgs.shape[0], -1)
print("Test Images Done")

init_path = '/groups/CS156b/data/'

solution_img_ids = []

j = 0
for i, row in solution.iterrows():
    if j >= num_test:
        break
    j += 1
    try:
        temp_img = asarray(Image.open(init_path + row['Path']))
        temp_img = cv2.resize(temp_img, (width, height))
        temp_img = temp_img/255
        solution_imgs.append(temp_img)
        solution_img_ids.append(row['Id'])
    except FileNotFoundError:
        print("Image doesn't exist")
        
        
solution_imgs = np.array(solution_imgs)
reshaped_solution_imgs = solution_imgs.reshape(solution_imgs.shape[0], -1)
print("Solution Images Done")


np.savetxt('/central/groups/CS156b/teams/clnsh/subset_train_matrices', reshaped_train_imgs)
np.savetxt('/central/groups/CS156b/teams/clnsh/subset_test_matrices', reshaped_test_imgs)
np.savetxt('/central/groups/CS156b/teams/clnsh/subset_solution_matrices', reshaped_solution_imgs)
np.savetxt('/central/groups/CS156b/teams/clnsh/subset_train_labels', train_labels)
np.savetxt('/central/groups/CS156b/teams/clnsh/subset_test_img_ids', test_img_ids, fmt = '%s')
np.savetxt('/central/groups/CS156b/teams/clnsh/subset_solution_img_ids', solution_img_ids, fmt = '%s')


# In[56]:


# To load these files and convert them back to a 3d list:
# https://www.geeksforgeeks.org/how-to-load-and-save-3d-numpy-array-to-file-using-savetxt-and-loadtxt-functions/

# train_loaded = np.loadtxt('train_matrices')
# train_3d = train_loaded.reshape(num_train, (width*height) // width, width)

# test_loaded = np.loadtxt('test_matrices')
# test_3d = test_loaded.reshape(num_test, (width*height) // width, width)

# solution_loaded = np.loadtxt('solution_matrices')
# solution_3d = solution_loaded.reshape(MAX_TEST, (width*height) // width, width)

# train_loaded = np.loadtxt('train_labels')
# test_img_ids = np.loadtxt('test_img_ids')
# solution_img_ids = np.loadtxt('solution_img_ids')

