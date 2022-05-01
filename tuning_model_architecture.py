import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt #!pip install keras-tuner -q
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


''' 
Model builds with hyperparameters, which is defining the search space
'''

def model_build_default(hyper_param):
    model = keras.Sequential()
    model.add(layers.Flatten())
    
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
    
    print(hyper_param .Int("units", min_value=32, max_value=512, step=32))
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

    print(hyper_param .Int("units", min_value=32, max_value=512, step=32))
    return model

''' 
Starting the search and importing the data. 
Possible tuner classes: RandomSearch, Hyperband, BayesianOptimization
'''
def set_up_tuner(tuner_class, objective, maxt, executions, over_write, direct, proj_name):
# initialization of tuner, need to fill in directory and project_name depedning on who is running program
    tuner = kt.tuner_class(
        hypermodel = model_build_boolean, 
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













