from keras.models import Sequential, load_model
from keras.activations import relu
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from keras import losses
from keras import regularizers
from keras.initializers import RandomUniform

initializer = RandomUniform(seed=1234)

def getModel(id, input_dim):

    print(">>> Creating model...")
    model = Sequential()
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        RETI USATE    <<<<<<<<<<<<<<<<<<<<<<<<<            

    if id=="best_model":
        model.add(Dense(units=512, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=256,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=128,activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation = "sigmoid"))

        model.compile(loss=losses.binary_crossentropy,
                    optimizer='adam',
                    metrics = ['accuracy'])
    
        print(model.summary())

    elif id=="regularized_best_model_l1":
        model.add(Dense(units=512, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=256,activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(units=128,activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation = "sigmoid"))

        model.compile(loss=losses.binary_crossentropy,
                    optimizer='adam',
                    metrics = ['accuracy'])
    
        print(model.summary())


    elif id=="regularized_best_model_l2":
        model.add(Dense(units=512, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=256,activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(units=128,activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation = "sigmoid"))

        model.compile(loss=losses.binary_crossentropy,
                    optimizer='adam',
                    metrics = ['accuracy'])
    
        print(model.summary())

    elif id=="no_regularization":
        model.add(Dense(units=512, input_dim=input_dim, activation="relu"))
        model.add(Dense(units=256,activation="relu"))
        model.add(Dense(units=128,activation="relu"))

        model.add(Dense(2, activation = "sigmoid"))

        model.compile(loss=losses.binary_crossentropy,
                    optimizer='adam',
                    metrics = ['accuracy'])
    
        print(model.summary())

    
    return model