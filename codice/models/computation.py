'''
Excecute the computation function of the whole project
'''

import argparse
import pandas as pd
from preprocessing.preprocessing import preprocessing_data
from preprocessing.explorative_plots import loss_plotter
from preprocessing.explorative_plots import scatter_plotter
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, zscore
import statistics
import seaborn as sns
import joblib
import argparse
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn import metrics


from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
history = History()
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import metrics
from models.all_models import getModel

pd.options.mode.chained_assignment = None



def computation(args):
    '''
    Function that execute in sequence the all operations of the project, saving partial and final results.
    The steps are: preprocessing, explorative plots, modeling and results
    Parameters:
        args    - Required  : command line args 
    '''
    if tf.test.gpu_device_name(): 
        print('Using default GPU device:{}'.format(tf.test.gpu_device_name())) 
    else: 
        print("Not using GPU.")

    if not os.path.exists('../data/scaled'):
        os.mkdir('../data/scaled')

    if not os.path.exists('../data/scaled/x_train_scaled.npy'):
        x_train_scaled, x_validation_scaled, x_test_scaled, y_train_scaled, y_validation, y_validation_scaled = preprocessing_data()
        np.save(arr = x_train_scaled, file = '../data/scaled/x_train_scaled.npy')
        np.save(arr = x_validation_scaled, file = '../data/scaled/x_validation_scaled.npy')
        np.save(arr = x_test_scaled, file = '../data/scaled/x_test_scaled.npy')
        np.save(arr = y_train_scaled, file = '../data/scaled/y_train_scaled.npy')
        np.save(arr = y_validation, file = '../data/scaled/y_validation.npy')
        np.save(arr = y_validation_scaled, file = '../data/scaled/y_validation_scaled.npy')
    else:
        x_train_scaled = np.load('../data/scaled/x_train_scaled.npy')
        x_validation_scaled = np.load('../data/scaled/x_validation_scaled.npy')
        x_test_scaled = np.load('../data/scaled/x_test_scaled.npy')
        y_train_scaled = np.load('../data/scaled/y_train_scaled.npy')
        y_validation = np.load('../data/scaled/y_validation.npy')
        y_validation_scaled = np.load('../data/scaled/y_validation_scaled.npy')
    
    print(f"Train shape: x:{x_train_scaled.shape}, y:{y_train_scaled.shape}")
    print(f"Validation shape: x:{x_validation_scaled.shape}, y:{y_validation_scaled.shape}")
    print(f"Test shape: x:{x_test_scaled.shape}")

    if args.model == None:
        model = getModel(args.model_schema, x_train_scaled.shape[1])
    else:
        print(">>> Loading model ({0})...".format(args.model))
        model = load_model(args.model)


    if not args.evaluate:
        # Training procedure
        
        '''if args.save_steps:
            auto_save = ModelCheckpoint(args.output+"/current_model_epoch{epoch:02d}", monitor='val_loss',
                        verbose=0, save_best_only=False, save_weights_only=True,
                        mode='auto', period=1)
        else:
            auto_save = ModelCheckpoint(args.output +"/current_model", monitor='val_loss',
                        verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=2)
        '''

        if args.save_steps:
            auto_save = ModelCheckpoint(args.output+"/current_model_epoch", monitor='val_loss',
                        verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto')
        else:
            auto_save = ModelCheckpoint(args.output +"/current_model", monitor='val_loss',
                        verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto')

        
        min_delta = float(args.patience.split(":")[0])
        p_epochs = int(args.patience.split(":")[1])
        early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta,
                                patience=p_epochs, verbose=0)


        def lr_scheduler_fede(epoch, lr):
            if epoch <= 7:
                lr = args.learning_rate
            if (epoch > 7) & (epoch <= 12):
                lr = 0.001
            if (epoch > 12) & (epoch <= 20):
                lr = 0.0007
            if epoch > 20:
                lr = 0.0005
            return lr

     
        lr_sched = LearningRateScheduler(lr_scheduler_fede, verbose=1)

        csv_logger = CSVLogger(args.output +'/training.log')


        print(">>> Training...")

        # Cambiare inizializzazione dei pesi
        W_val = 0.5 * np.random.randn(x_train_scaled.shape[1]) + 1

        # Fitting
        history = model.fit(x_train_scaled, y_train_scaled,
                            validation_data = (x_validation_scaled, y_validation_scaled),
                            epochs=args.epoch, initial_epoch=args.initial_epoch,
                            batch_size=args.batch_size, shuffle=True,
                            callbacks=[early_stop, lr_sched, csv_logger, auto_save])
        

    ################## COMPUTE THE CONFUSION MATRIX AND THE LOSS ON THE VALIDATION #######
    loss_plotter(history, args.output)

    scaler_y = joblib.load("scaler_y.pkl")
    # Importo lo scaler per usare al contrario le predizioni
    model.load_weights(args.output+"/current_model")
    y_validation_scaled_pred = model.predict(x_validation_scaled)
    y_validation_pred =  scaler_y.inverse_transform(y_validation_scaled_pred)
    scatter_plotter(y_validation, y_validation_pred, args.output)
    np.save(arr = y_validation_pred, file = args.output+'/y_val_pred.npy')
    np.save(arr = y_validation_scaled_pred, file = args.output+'/y_validation_scaled_pred.npy')
    

    ################# COMPUTE THE PREDICTION ON THE TEST #############
    y_test_scaled_pred = model.predict(x_test_scaled)
    y_test_pred =  scaler_y.inverse_transform(y_test_scaled_pred)
    np.save(arr = y_test_pred, file = args.output+'/y_test_pred.npy')
    

    ################# SAVE THE WEIGHTS ###################À
    if not args.evaluate:    
        print(">>>>>>>>> SAVING WEIGHTS >>>>>>>>")
        f = open(args.output + "/weights.txt", "w")
        f.write('Layers name: {}\n'.format(model.weights[-2].name))
        f.write('Layers kernel shape: {}\n'.format(model.weights[-2].shape))
        f.write('Kernel: {}\n'.format(model.weights[-2][0], end = '\n\n'))
        f.write('Layers name: {}\n'.format(model.weights[-1].name))
        f.write('Layers kernel shape: {}\n'.format(model.weights[-1].shape))
        f.write('Kernel: {}\n'.format(model.weights[-1]))
        f.close()
    