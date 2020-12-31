'''
    Program that opens the files before preprocessing
'''

import pandas as pd

def open_data():
    '''
    Function that load data
    '''


    train = pd.read_csv("../data/train.csv", sep = ",")
    test = pd.read_csv("../data/test.csv", sep = ",")

    return train, test
