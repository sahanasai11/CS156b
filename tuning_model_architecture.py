import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt #!pip install keras-tuner -q
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


''' 
Model builds with hyperparameters, which is defining the search space
'''


def model_build_curr(hyper_param):
    # our current model 
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(100,100,1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3,3)))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024,activation='relu'))
    model.add(keras.layers.Dense(1024,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1024,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(14,activation='sigmoid'))


    # tuning the units 
    # an integer hyperparameter with an inclusive range from 32 to 512 
    # Also, going through the inteveral has a minimum step of 32 
    first_hp = hyper_param.Int('units', min_value=32, max_value=512, step=32)

    #activation parameter
    # default is relu 
    default_activation = 'relu'
 
    #for simplicity, we are going to add two Dense layers while tuning the number of units 
    # in the first layer 
    model.add(layers.Dense(units=first_hp, activation='relu'))
    # change the parameters of the model to correspond with corresponding model 
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(hyper_param.Int("units", min_value=32, max_value=512, step=32))
    return model


def model_build_boolean(hyper_param):
    model = keras.Sequential()
    model.add(layers.Flatten())
    
    # tuning the units 
    # an integer hyperparameter with an inclusive range from 32 to 512 
    # Also, going through the inteveral has a minimum step of 32 
    first_hp = hyper_param.Int('units', min_value=32, max_value=512, step=32)

    #activation parameter
    # tuning which activation function to use
    activation_level = hyper_param.Choice('activation', ['relu', 'tanh'])
   
    model.add(layers.Dense(units=first_hp, activation=activation_level))

    # use boolean proccess to tune with ot without a Dropout Layer 
    drate = .25
    if hyper_param.Boolean('dropout'):
        model.add(layers.Dropout(rate=drate))
    
    # continue to add softmax layer 
    model.add(layers.Dense(10, activation='softmax'))

    # tun the optimizer learning rate (i.e. define as a hyperparameter)
    lrate = hyper_param.Float('lr', min_value = .00001, max_value= .01, sampling='log')
    optimizer_lr = keras.optimizers.Adam(learning_rate=lrate)
    model.compile(optimizer=optimizer_lr, loss='categorical_crossentropy', metrics=['accuracy'])

    print(hyper_param.Int("units", min_value=32, max_value=512, step=32))
    return model

''' 
Starting the search and importing the data. 
Possible tuner classes: RandomSearch, Hyperband, BayesianOptimization
'''
def set_up_tuner(objective, maxt, executions, over_write, direct, proj_name):
# initialization of tuner, need to fill in directory and project_name depedning on who is running program
    tuner = kt.RandomSearch(
        hypermodel = model_build_curr, 
        objective = objective, #'val_accuracy'
        max_trials = maxt, 
        executions_per_trial=executions, 
        overwrite=over_write, #boolean
        directory=direct, 
        project_name=proj_name
    )

    tuner.search_space_summary()
    return tuner

def run_tuner(tuner, x_train, y_train, eps, x_validation, y_validation, no_models, x, y, z):
    # searches for the best hyperparameter configuration 
    # note, all values passed into search are also passed into model.fit()
    tuner.search(x_train, y_train, epochs=eps, validation_data=(x_validation, y_validation))
    tuner.results_summary() 

    # get results 
    final_models = tuner.get_best_models(num_models=no_models)
    best = final_models[0]

    # build the best model 
    best.build(input_shape=(x, y, z))
    best.summary()

    # logs are located in directory/project_name


'''
Rebuilding the model with the best hyperparameters
'''
def retrain_model(tuner, x_train, y_train, eps, x_validation, y_validation):
    # get the top 2 params
    best_params = tuner.get_best_hyperparameters(5)
    model = model_build_boolean(best_params[0])

    # have to fit the entire dataset 
    x_new = np.concatenate((x_train, x_validation))
    y_new = np.concatenate((y_train, y_validation))

    # epochs should be 1 mostly
    model.fit(x=x_new, y=y_new, epochs=eps)
    return model


MAX_TRAIN = 178158
MAX_TEST = 22596

SUBSET_TRAIN = 2000
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
#ROOT_PATH = '/central/groups/CS156b/teams/clnsh/compressed_data/'

# For subset data use: 
ROOT_PATH = '/central/groups/CS156b/teams/clnsh/subset_compressed_data10/subset_'
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

print('test id loaded ', len(test_img_id))
#print('test, solution ids loaded: ', len(test_img_id), ',', len(solution_img_id))

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



xx_train = X_train[:-100000]
yy_train = y_train[:-100000]
xx_val = X_train[-100000:]
yy_val = y_train[-100000:]


tuner = set_up_tuner('val_accuracy', 3, 2,  True, '/home/hsiaveli/ondemand/data/sys/dashboard/batch_connect/sys/jupyterlab/output', 'tuner_creation')
# implementing the hyperparameters 
# tuning class 
run_tuner(tuner, X_train, y_train, 10, xx_train, yy_train, xx_val, yy_val)












