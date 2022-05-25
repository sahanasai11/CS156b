# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# import np_utils # pip install np-utils

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


# In[2]:


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

print('train images loaded: ', len(train))

#used to be train_path + train_labels
train_label = pd.read_table(TRAIN_LABELS_PATH + 'train_labels_nan', delimiter=" ", 
              names=label_names)

print('train labels loaded: ', len(train_label))

# Testing data:
test_loaded = np.loadtxt(TEST_PATH + 'test_matrices') # NOTE: 224 is added
num_test = int(test_loaded.size / (width * height))
test = test_loaded.reshape(num_test, (width*height) // width, width)

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


# making 7 separate sub-arrays splitting the training lables
y_train_support = []
y_train_cardio = []
y_train_pneumothorax = []
y_train_p = []
y_train_fracture = []
y_train_other = []
y_train_nofinding = []

i = 0
while i < len(y_train_scaled):
    y_train_support.append(y_train_scaled[i][13]) # 0-indexed 13th element
    y_train_cardio.append(y_train_scaled[i][1:3]) # 0-indexed 1st - 2nd element
    y_train_pneumothorax.append(y_train_scaled[i][9]) # 0-indexed 9th element
    y_train_p.append(y_train_scaled[i][10:12]) # 0-indexed 10th - 11th element
    y_train_fracture.append(y_train_scaled[i][12]) # 0-indexed 12th element
    y_train_nofinding.append(y_train_scaled[i][0]) # 0-indexed 0th element
    y_train_other.append(y_train_scaled[i][3:9]) # 0-indexed 3rd - 8th element
    i += 1

y_train_support = np.array(y_train_support)
y_train_cardio = np.array(y_train_cardio)
y_train_p = np.array(y_train_p)
y_train_fracture = np.array(y_train_fracture)
y_train_nofinding = np.array(y_train_nofinding)
y_train_other = np.array(y_train_other)
y_train_pneumothorax = np.array(y_train_pneumothorax)

print('all labels [0:3]:')
print(y_train[0:3])
print('all labels scaled [0:3]:')
print(y_train_scaled[0:3])
print('support - 13th')
print(y_train_support[0:3])
print('cardio - 1st and 2nd')
print(y_train_cardio[0:3])
print('p - 9th, 10th, 11th element')
print(y_train_p[0:3])
print('fracture - 12th')
print(y_train_fracture[0:3])
print('no finding - 0th')
print(y_train_nofinding[0:3])
print('other - 3rd - 8th')
print(y_train_other[0:3])

print('y_train_support, cardio, p, fracture, nofinding, other: ', len(y_train_support), 
      len(y_train_cardio), len(y_train_p), len(y_train_fracture), len(y_train_nofinding), len(y_train_other))


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
# Pneumothorax
# -----------------------------------------------------------------------

X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
test = np.repeat(test[..., np.newaxis], 3, -1)

tensor_train2 = []
tensor_test2 = []

for i in range(0, len(X_train)):
    tensor_train2.append(tf.convert_to_tensor(X_train[i], dtype=float))
print('train tensors created')

for i in range(0, len(test)):
    tensor_test2.append(tf.convert_to_tensor(test[i], dtype=float))
print('test tensors created')

# for i in range(0, len(solution)):
#     tensor_solution.append(tf.convert_to_tensor(solution[i], dtype=float))
# print('solution tensors created')

# setting training and testing data to be a tensor of tensor-ed images
# shape: n training images, shape width, shape height
train_tensor_of_tensors2 = tf.convert_to_tensor(tensor_train2)
print('train tensor of tensor created:', train_tensor_of_tensors2.shape)

test_tensor_of_tensors2 = tf.convert_to_tensor(tensor_test2)
print('test tensors of tensors created:', test_tensor_of_tensors2.shape)


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
    
history_pneumothorax = model_pneumothorax.fit(train_tensor_of_tensors2, y_train_scaled, epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_pneumothorax = model_pneumothorax.predict(test_tensor_of_tensors2)
preds_pneumothorax = preds_pneumothorax.tolist()
print('pneumothorax predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# BINARY, SUPPORT DEVICES 
# -----------------------------------------------------------------------
num_batch = 100
num_epoch = 15
num_class = 1

model_support = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(width, height, 1)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(num_class, activation='sigmoid')])

model_support.compile(loss='binary_crossentropy',
optimizer=tf.optimizers.Adam(learning_rate=0.005),
metrics='mean_squared_error')
    
history_support = model_support.fit(train_tensor_of_tensors, y_train_support, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_support = model_support.predict(test_tensor_of_tensors)
preds_support = preds_support.tolist()
print('support device predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# BINARY, NO FINDING
# -----------------------------------------------------------------------
num_batch = 100
num_epoch = 15
num_class = 1

model_nofinding = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(width, height, 1)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(num_class, activation='sigmoid')])

model_nofinding.compile(loss='binary_crossentropy',
optimizer=tf.optimizers.Adam(learning_rate=0.005),
metrics='mean_squared_error')
    
history_nofinding = model_nofinding.fit(train_tensor_of_tensors, y_train_nofinding, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_nofinding = model_nofinding.predict(test_tensor_of_tensors)
preds_nofinding = preds_nofinding.tolist()
print('no finding predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# BINARY, FRACTURE
# -----------------------------------------------------------------------
num_batch = 100
num_epoch = 15
num_class = 1

model_frac = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(width, height, 1)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(num_class, activation='sigmoid')])

model_frac.compile(loss='binary_crossentropy',
optimizer=tf.optimizers.Adam(learning_rate=0.005),
metrics='mean_squared_error')
    
history_frac = model_frac.fit(train_tensor_of_tensors, y_train_fracture, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_frac = model_frac.predict(test_tensor_of_tensors)
preds_frac = preds_frac.tolist()
print('fracture predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# ALEX NET, ENLARGED CARDIOM., CARDIOMEGALY
# -----------------------------------------------------------------------
num_batch = 100
num_epoch = 15
num_class = 2

model_cardio = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(width,height,1)),
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
    keras.layers.Dense(num_class,activation='sigmoid')])

model_cardio.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])

history_cardio = model_cardio.fit(train_tensor_of_tensors, y_train_cardio, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_cardio = model_cardio.predict(test_tensor_of_tensors)
preds_cardio = preds_cardio.tolist()

print('cardio predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# ALEX NET Pleural Effusion, Pleural Other
# -----------------------------------------------------------------------
num_batch = 64
num_epoch = 15
num_class = 2

model_p = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(width,height,1)),
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
    keras.layers.Dense(num_class,activation='sigmoid')])

model_p.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])

history_p = model_p.fit(train_tensor_of_tensors, y_train_p, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds_p = model_p.predict(test_tensor_of_tensors)
preds_p = preds_p.tolist()

print('p predictions complete')


# In[ ]:


# -----------------------------------------------------------------------
# ALEX NET, 6 CLASSES
# -----------------------------------------------------------------------
num_batch = 64
num_epoch = 15
num_class = 6

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(width,height,1)),
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
    keras.layers.Dense(num_class,activation='sigmoid')])

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error'])


history = model.fit(train_tensor_of_tensors, y_train_other, validation_split = 0.2, 
                    epochs=num_epoch, verbose=1, batch_size=num_batch)

preds = model.predict(test_tensor_of_tensors)
preds = preds.tolist()

print('6 class predictions complete')


# In[ ]:


combined = copy.deepcopy(preds)

for i in range(len(tensor_test)):
    img_id = test_img_id[i]
    
    combined[i].insert(0, preds_nofinding[i][0])
    
    combined[i].insert(1, preds_cardio[i][1])
    combined[i].insert(1, preds_cardio[i][0])
    
    combined[i].insert(len(combined) - 3, preds_pneumothorax[i][0])
    combined[i].insert(len(combined) - 3, preds_p[i][0])
    combined[i].insert(len(combined) - 3, preds_p[i][1])
    
    combined[i].append(preds_frac[i][0])
    combined[i].append(preds_support[i][0])
    
    combined[i].insert(0, img_id)

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


# In[19]:


# -----------------------------------------------------------------------
# LOSS PLOTTING
# -----------------------------------------------------------------------

plt.plot(history_support.history['loss'])
plt.plot(history_support.history['val_loss'])
plt.title('support devices model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('support_model_loss.png')
plt.show()

plt.plot(history_cardio.history['loss'])
plt.plot(history_cardio.history['val_loss'])
plt.title('cardio model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cardio_model_loss.png')
plt.show()

plt.plot(history_pneumothorax.history['loss'])
plt.plot(history_pneumothorax.history['val_loss'])
plt.title('pneumothorax model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('pneumothorax_model_loss.png')
plt.show()

plt.plot(history_p.history['loss'])
plt.plot(history_p.history['val_loss'])
plt.title('p model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('p_model_loss.png')
plt.show()

plt.plot(history_frac.history['loss'])
plt.plot(history_frac.history['val_loss'])
plt.title('fracture model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('fracture_model_loss.png')
plt.show()

plt.plot(history_nofinding.history['loss'])
plt.plot(history_nofinding.history['val_loss'])
plt.title('no finding model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('no_finding_model_loss.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('other classification model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('other_model_loss.png')
plt.show()
