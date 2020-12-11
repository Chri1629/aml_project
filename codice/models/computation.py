import argparse
import pandas as pd
from preprocessing.preprocessing import preprocessing_data
from preprocessing.explorative_plots import loss_plotter
from preprocessing.explorative_plots import scatter_plotter
from keras import *
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
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn import metrics


from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras import losses
from keras.callbacks import EarlyStopping, History
from keras.activations import relu
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
history = History()
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.activations import relu
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from .all_models import getModel

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, required=False, default=5, help ="Give the number of epochs, default = 10")
parser.add_argument('-ie', '--initial_epoch', type=int, required=False, default=0, help ="Give the initial epoch, default = 0")
parser.add_argument('-bs', '--batch-size', type=int, required=False, default=128, help ="Give the batch size, default = 128")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=5e-3, help ="Give the learning rate, default = 1e-3") 
parser.add_argument('-d', '--data', type=str, required=False, default="data", help = "Give the directory of the data")
parser.add_argument('-p', '--patience', type=str, required=False, default="0.01:50",
                    help="Patience format:  delta_val:epochs")
parser.add_argument('-m', '--model', type=str, required=False)
parser.add_argument('-ev', '--evaluate', action="store_true")
parser.add_argument('-sst', '--save-steps', action="store_true")
parser.add_argument('-dr', '--decay-rate', type=float, required=False, default=0, help ="Give the decay rate")
parser.add_argument('-ms', '--model-schema', type=str, required=False, default = "best_model", help="Model structure")
parser.add_argument('-o', '--output', type=str, required=False, default="risultati_modelli")

args = parser.parse_args()
pd.options.mode.chained_assignment = None

def computation():
    x_train_scaled, x_validation_scaled, x_test_scaled, y_train_scaled, y_validation, y_validation_scaled = preprocessing_data()

    if args.model == None:
        model = getModel(args.model_schema, x_train_scaled.shape[1])
    else:
        print(">>> Loading model ({0})...".format(args.model))
        model = load_model(args.model)


    if not args.evaluate:
        # Training procedure
        if args.save_steps:
            auto_save = ModelCheckpoint(args.output+"/current_model_epoch{epoch:02d}", monitor='val_loss',
                        verbose=0, save_best_only=False, save_weights_only=False,
                        mode='auto', period=1)
        else:
            auto_save = ModelCheckpoint(args.output +"/current_model", monitor='val_loss',
                        verbose=0, save_best_only=True, save_weights_only=False,
                        mode='auto', period=2)

        min_delta = float(args.patience.split(":")[0])
        p_epochs = int(args.patience.split(":")[1])
        early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta,
                                patience=p_epochs, verbose=0)

        def reduceLR (epoch):
            return args.learning_rate * (1 / (1 + epoch*args.decay_rate))

        lr_sched = LearningRateScheduler(reduceLR, verbose=0)

        csv_logger = CSVLogger(args.output +'/training.log')


        print(">>> Training...")

        # Cambiare inizializzazione dei pesi
        W_val = 0.5 * np.random.randn(x_train_scaled.shape[1]) + 1

        history = model.fit(x_train_scaled, y_train_scaled,
                            validation_data = (x_validation_scaled, y_validation_scaled),
                            epochs=args.epoch, initial_epoch=args.initial_epoch,
                            batch_size=args.batch_size, shuffle=True,
                            callbacks=[auto_save, early_stop, lr_sched, csv_logger])
        

    ################## COMPUTE THE CONFUSION MATRIX AND THE LOSS ON THE VALIDATION #######
    loss_plotter(history)

    scaler_y = joblib.load("scaler_y.pkl")
    # Importo lo scaler per usare al contrario le predizioni
    y_validation_scaled_pred = model.predict(x_validation_scaled)
    y_validation_pred =  scaler_y.inverse_transform(y_validation_scaled_pred)
    scatter_plotter(y_validation, y_validation_pred)
    

    ################# COMPUTE THE PREDICTION ON THE TEST #############
    y_test_scaled_pred = model.predict(x_test_scaled)
    y_test_pred =  scaler_y.inverse_transform(y_test_scaled_pred)


    ################# SAVE THE WEIGHTS ###################À
    #if not args.evaluate:    
    #    print(">>>>>>>>> SAVING WEIGHTS >>>>>>>>")
    #    f = open(args.output + "/weights.txt", "w")
    #    f.write('Layers name: {}\n'.format(model.weights[-2].name))
    #    f.write('Layers kernel shape: {}\n'.format(model.weights[-2].shape))
    #    f.write('Kernel: {}\n'.format(model.weights[-2][0], end = '\n\n'))
    #    f.write('Layers name: {}\n'.format(model.weights[-1].name))
    #    f.write('Layers kernel shape: {}\n'.format(model.weights[-1].shape))
    #    f.write('Kernel: {}\n'.format(model.weights[-1]))
    #    f.close()