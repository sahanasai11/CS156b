
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import matplotlib.pyplot as plt

import copy

import cv2

train = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')


# In[11]:


train_labels = []
num_train = len(train)
j = 0
for index, row in train.iterrows():
    if j % 10000 == 0:
        print(j / num_train)
    j += 1
    try:
        # training labels
        temp2 = [row['No Finding'], row['Enlarged Cardiomediastinum'], row['Cardiomegaly'],
            row['Lung Opacity'], row['Lung Lesion'], row['Edema'], row['Consolidation'],
            row['Pneumonia'], row['Atelectasis'], row['Pneumothorax'], row['Pleural Effusion'], 
            row['Pleural Other'], row['Fracture'], row['Support Devices']]
        i = 0 
        for val in temp2: 
            if val != val: # Handles NaN's
                temp2[i] = 10.0
            i += 1
        train_labels.append(temp2)
        
    except FileNotFoundError:
        print("Image doesn't exist")


# In[31]:


plt.hist([i[4] for i in train_labels])
plt.title('Lung Opacity')


# In[32]:


plt.hist([i[11] for i in train_labels])
plt.title('Pleural Other')

