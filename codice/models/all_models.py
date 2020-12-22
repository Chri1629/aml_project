from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import regularizers
#from keras.initializers import RandomUniform

# initializer = RandomUniform(seed=1234)

def getModel(id, input_dim):

    print(">>> Creating model...")
    model = Sequential()
# <<<<<<<<<<<<<<<<<<<<<<<<<     RETI     <<<<<<<<<<<<<<<<<<<<<<<<<            

    if id=="best_model":
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

    elif id=="fede1":
        model.add(Dense(units=1024, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(units=512,activation="relu"))
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
    
    return model