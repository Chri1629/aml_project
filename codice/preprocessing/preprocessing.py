from .open import open_data
from .explorative_plots import histogram_plot

def preprocessing_data():
    train, test = open_data()
    print("************* TRAIN DATA ***************\n", train.head())
    print()
    print("************* TEST DATA ***************\n",test.head())