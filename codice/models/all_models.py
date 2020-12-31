'''
Program to import the requested neural network for computation
'''

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout,AlphaDropout, Conv1D, Flatten, BatchNormalization, Input, GaussianNoise, ReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras.applications import resnet50, vgg19, mobilenet_v2, xception, resnet


def getModel(id, input_dim):
    '''
    Function to load the requested model
    @params:
        id:        - Required   : The id of the requested model
        input_dim:   - Required   : The input dimensions for the model
    '''

    print(">>> Creating model...")
    model = Sequential()
# <<<<<<<<<<<<<<<<<<<<<<<<<     RETI     <<<<<<<<<<<<<<<<<<<<<<<<<            

    if id=="model1": 
        model.add(Input(shape=input_dim))
        model.add(GaussianNoise(0.005))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])

        print(model.summary())


    elif id=="model2": 
        model.add(Input(shape=input_dim))
        model.add(GaussianNoise(0.005))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="model3": 
        model.add(Input(shape=input_dim))
        model.add(GaussianNoise(0.005))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())
    

    return model

''' elif id=="fede1": #NO
        model.add(Dense(units=512, input_dim=input_dim,activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(units=512,activation="relu"))        
        model.add(Dropout(0.4))
        model.add(Dense(units=256,activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(units=256,activation="relu"))        
        model.add(Dropout(0.4))

        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())


    elif id=="best_model":
        model.add(Dense(units=128, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=64,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=32,activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="regularized_best_model_l1":
        model.add(Dense(units=512, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=256,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=256,activation="relu"))        
        model.add(Dropout(0.2))

        model.add(Dense(2, activation = "sigmoid"))

        model.compile(loss=losses.binary_crossentropy,
                    optimizer='adam',
                    metrics = ['accuracy'])
    
        print(model.summary())  

    elif id=="fede4": 
        model.add(Dense(units=64, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())
    

    elif id=="fede4-2": 
        model.add(Input(shape=input_dim))
        model.add(GaussianNoise(0.005))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])

        print(model.summary())

    elif id=="fede5": #NO
        model.add(Dense(units=64, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.5))
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="fede2": #NO
        model.add(Dense(units=150, input_dim=input_dim,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=70,activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(units=30,activation="relu"))        
        model.add(Dropout(0.4))

        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="fede3": #NO
        model.add(Dense(units=128, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=64,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=32,activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="fede6":
        model.add(Dense(units=128, input_dim=input_dim, activation="relu", activity_regularizer=regularizers.l1(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="fede6-1":
        model.add(Dense(units=128, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="fede6-2": #NO
        model.add(Dense(units=128, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))        
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

    elif id=="fede6-3": #NO
        model.add(Dense(units=128, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l1_2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())
    


    elif id=="fede8": #NO
        model.add(Dense(units=256, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)))
        model.add(BatchNormalization())
        model.add(Dense(1))

        model.compile(loss=losses.mean_squared_error,
                    optimizer='adam',
                    metrics = ['mse'])
    
        print(model.summary())

'''

    