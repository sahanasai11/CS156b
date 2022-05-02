
# coding: utf-8

# In[1]:


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

SUBSET_TRAIN = 2000
SUBSET_TEST = 700

num_train = SUBSET_TRAIN
num_test = SUBSET_TEST

width = 100
height = 100           
     
label_names = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
               "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia",
               "Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other",
               "Fracture","Support Devices"]

id_header = 'Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices'

# For full data use:
# ROOT_PATH = '/central/groups/CS156b/teams/clnsh/compressed_data/'

# For subset data use: 
ROOT_PATH = '/central/groups/CS156b/teams/clnsh/subset_compressed_data/subset_'

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


# In[4]:


# -----------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------

num_classes = 14
num_batch = 32
num_epoch = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=[width, height, 1]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mean_squared_error'])

history = model.fit(train_tensor_of_tensors, y_train, epochs=num_epoch, verbose=1, batch_size=num_batch)

print('model fit complete')


preds = model.predict(test_tensor_of_tensors)
preds = preds.tolist()

print('predictions complete')

scaled_preds = []
for i in range(len(preds)):
    curr = np.array(preds[i])
    factor = min(curr)
    newList = [(x - factor) for x in curr]
    max_curr = max(newList)
    myInt = max_curr / 2
    newList2 = [((x / myInt)-1) for x in newList]
    scaled_preds.append(newList2)

print('scaled predictions complete')


for i in range(len(tensor_test)):
    img_id = test_img_id[i]
    scaled_preds[i].insert(0, img_id)

np.set_printoptions(suppress=True) # Stop ID's from being converted to scientific notation
np.savetxt("test_sub.csv", scaled_preds, delimiter=",", fmt='%f', header = id_header, comments = '')
print('hi')


# In[26]:


#converting = keras.utils.to_categorical(y_train, 14)
#X_p = keras.applications.densenet.preprocess_input(train)
#X_p.shape
arr_3d = train.reshape((num_train, 100,100,3))


# In[4]:


import tensorflow as tf
import tensorflow.keras as K
initializer = K.initializers.he_normal(seed=32)

input_shape_densenet = (width, height,3)

densenet_model = DenseNet169(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape_densenet,
    pooling=None
)


densenet_model.trainable = True

for layer in densenet_model.layers:
  if 'conv5' in layer.name:
    layer.trainable = True
  else:
    layer.trainable = False

#densenet_model.summary()

input = K.Input(shape=(width, height))
preprocess = K.layers.Lambda(lambda x: tf.image.resize(x, (width, height)), name='lamb')(input)

layer = densenet_model(inputs=preprocess)

layer = K.layers.Flatten()(layer)

layer = K.layers.BatchNormalization()(layer)

layer = K.layers.Dense(units=256,
                        activation='relu',
                        kernel_initializer=initializer
                        )(layer)

layer = K.layers.Dropout(0.4)(layer)

layer = K.layers.BatchNormalization()(layer)

layer = K.layers.Dense(units=128,
                       activation='relu',
                       kernel_initializer=initializer
                       )(layer)

layer = K.layers.Dropout(0.4)(layer)

layer = K.layers.Dense(units=10,
                       activation='softmax',
                       kernel_initializer=initializer
                       )(layer)

model = K.models.Model(inputs=input, outputs=layer)

model.summary()

model.compile(loss='binary_crossentropy',
            optimizer=K.optimizers.Adam(),
            metrics=['accuracy'])


history = model.fit(train_tensor_of_tensors, y_train, epochs=10, batch_size=32, verbose=1)


# In[5]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


num_classes = 14
num_batch = 100
num_epoch = 10

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
    keras.layers.Dense(14,activation='sigmoid')  
    
    
])


# In[8]:


model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=['mean_squared_error']    
)


# In[9]:


model.summary()


# In[10]:


history = model.fit(train_tensor_of_tensors, y_train, epochs=num_epoch, verbose=1, batch_size=num_batch)


# In[13]:


y_train.shape

