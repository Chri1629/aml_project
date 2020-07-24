'''
    Programma che apre tutti i file e ne fa un primo preprocessing
'''

import pandas as pd
from .explorative_plots import histogram_plot

def open_data():
    '''
    Funzione che carica preliminarmente i dataset e li esplora
    '''

    # Change the path to the dataset
    path = "/home/christian/Scrivania/progetti/aml_project/data/"

    #sample = pd.read_csv("data/sample_submission_V2.csv", sep = ",")
    train = pd.read_csv(path + "train_V2.csv", sep = ",")
    test = pd.read_csv("data/test_V2.csv", sep = ",")

    histogram_plot(train['matchDuration'], test['matchDuration'])