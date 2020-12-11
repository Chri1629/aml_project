from .open import open_data
from .explorative_plots import histo_plot
from .explorative_plots import target_distribution
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import pandas as pd
import os

def check_missing(df):
    for colonna in df.columns:
        perc_missing = round(df[colonna].isnull().sum(axis = 0)/len(df[colonna])*100,0)
        if perc_missing == 0:
            pass
        else:
            print("La percentuale di nulli in", colonna, "è: ", perc_missing, "%", "in", df)

def delete_outliers(df_train):
    # Visto che è un'operazione molto lunga salviamo i dati senza outlier, così da fare una sola volta e controllare. L'unica
    # colonna che contiene outlier sembra l'ultima, facciamolo solo su quella. Poi cambiamo il path per non farglielo fare

    if not os.path.exists('../data/x_train_no_out.csv'):
        # Prima elimino la data perché fa casino per eliminare out
        df_train_input = df_train.iloc[:,0:(len(df_train.columns)-1)]
        df_train_target = df_train.iloc[:,-1]    
        target_no_out = df_train_target[np.abs(df_train_target-df_train_target.mean()) <= (3*df_train_target.std())]

        # Aggiungo nuovamente le altre colonne
        train_no_out = pd.merge(target_no_out, df_train_input, how = "inner", left_index=True, right_index=True)
        x_train_no_out = train_no_out.iloc[:,0:(len(train_no_out.columns)-1)]
        y_train_no_out = train_no_out.iloc[:,-1]

        # Salva in numpy array
        x_train_no_out.to_csv('../data/x_train_no_out.csv')
        y_train_no_out.to_csv('../data/y_train_no_out.csv')
        
    # Se i dati sono già presenti li carico
    x_train_no_out = pd.read_csv('../data/x_train_no_out.csv', sep = ",")
    y_train_no_out = pd.read_csv('../data/y_train_no_out.csv', sep = ",")
    return x_train_no_out, y_train_no_out


def preprocessing_data():
    train, test = open_data()
    print("************* TRAIN DATA ***************\n", train.head())
    print()

    print("************* TEST DATA ***************\n",test.head())
    print()

    print("************* CHECKING MISSING ON TRAINING ************ \n")
    check_missing(train)
    print()

    print("************* CHECKING MISSING ON TEST ************ \n")
    check_missing(test)
    print()

    print("************* DELETE EXTREME OBSERVATIONS ************ \n")
    # La variabile di target ha un bel po' di outlier quindi leviamoli
    x_train_no_out, y_train_no_out = delete_outliers(train)   
    print()

    print("************* SPLITTING DATA ************ \n")        
    x_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    x_test = test.copy()

    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.3, random_state=4242)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train_no_out, y_train_no_out, test_size=0.3, random_state=4242)

    print("************* PRODUCING PLOTS ************ \n")

    # Levo le date per disegnare e levo anche la data di dropoff nel train perché non è presente nel test
    train_for_distribution = x_train.drop(['pickup_datetime', 'dropoff_datetime'],axis =1)
    validation_for_distribution = x_validation.drop(['pickup_datetime', 'dropoff_datetime'],axis =1)
    test_for_distribution = x_test.drop('pickup_datetime', axis =1)
    
    histo_plot(train_for_distribution, validation_for_distribution, test_for_distribution)
    #target_distribution(y_train, y_validation)

    print("************* SCALE THE DATA ************ \n")
    # Scaliamo i dati ricordandoci di non scalare le date

    # Salviamo i dati in numpy array in modo da poterli riusare
    #np.save('../data/x_train_no_out.npy', x_train)
    #np.save('../data/y_train_no_out.npy', y_train)
    #np.save('../data/x_validation_no_out.npy', x_validation)
    #np.save('../data/y_validation_no_out.npy', y_validation)
    #np.save('../data/x_test.npy', x_test)

    return x_train, y_train, x_validation, y_validation, x_test