
# coding: utf-8

# In[2]:


get_ipython().system('pip3 install seaborn')

import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

width = 100
height = 100           
     
label_names = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
               "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia",
               "Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other",
               "Fracture","Support Devices"]

id_header = 'Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices'

ROOT_PATH = '/central/groups/CS156b/teams/clnsh/compressed_data/'

print('uploading data...')

train_label = pd.read_table(ROOT_PATH + 'train_labels', delimiter=" ", 
              names=label_names)

print('train labels loaded: ', len(train_label))

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

y_train = np.array(y_train)


# In[3]:


train_label.head()


# In[4]:


# heatmap of the correlation between the 14 classes

cols = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]

# create a dictionary where the keys are the labels, values are list of 1, 0, -1

class_dict = {}
for col in cols:
    class_dict[col] = []

# Splitting in batches because memory limit exceeds when doing all in a row
for index, row in train_label.iterrows():
    for col in range(3):
        class_dict[cols[col]].append(row[cols[col]])

for index, row in train_label.iterrows():
    for col in range(3, 6):
        class_dict[cols[col]].append(row[cols[col]])
        
for index, row in train_label.iterrows():
    for col in range(6, 9):
        class_dict[cols[col]].append(row[cols[col]])
        
for index, row in train_label.iterrows():
    for col in range(9, 12):
        class_dict[cols[col]].append(row[cols[col]])

for index, row in train_label.iterrows():
    for col in range(12, 14):
        class_dict[cols[col]].append(row[cols[col]])
        


# In[16]:


df = pd.DataFrame(class_dict, columns=cols)
corr_matrix = df.corr()
corr_matrix


# In[29]:


mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True


# In[31]:


f, ax = plt.subplots(figsize=(11, 15)) 
heatmap = sns.heatmap(corr_matrix, 
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 9})
#add the column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

