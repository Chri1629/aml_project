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
    #path = "/home/christian/Scrivania/progetti/aml_project/data/"

    train = pd.read_csv("../data/train.csv", sep = ",")
    test = pd.read_csv("../data/test.csv", sep = ",")

    return train, test
