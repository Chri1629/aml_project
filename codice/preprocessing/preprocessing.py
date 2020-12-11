from .open import open_data
from .explorative_plots import histo_plot
from .explorative_plots import target_distribution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy import stats
import pandas as pd
import haversine as hs
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

        #x_train_no_out = train_no_out.iloc[:,0:(len(train_no_out.columns)-1)]
        x_train_no_out = train_no_out.drop("trip_duration", axis = 1)
        y_train_no_out = train_no_out["trip_duration"]

        # Salva in numpy array
        x_train_no_out.to_csv('../data/x_train_no_out.csv')
        y_train_no_out.to_csv('../data/y_train_no_out.csv')
        
    # Se i dati sono già presenti li carico
    x_train_no_out = pd.read_csv('../data/x_train_no_out.csv', sep = ",", index_col=0)
    y_train_no_out = pd.read_csv('../data/y_train_no_out.csv', sep = ",", index_col=0)
    return x_train_no_out, y_train_no_out

def fix_lat_long(df):
    df['x_pickup'] = np.cos(df['pickup_latitude']) * np.cos(df['pickup_longitude'])
    df['y_pickup'] = np.cos(df['pickup_latitude']) * np.sin(df['pickup_longitude'])
    df['z_pickup'] = np.sin(df['pickup_latitude'])

    df['x_dropoff'] = np.cos(df['dropoff_latitude']) * np.cos(df['dropoff_longitude'])
    df['y_dropoff'] = np.cos(df['dropoff_latitude']) * np.sin(df['dropoff_longitude'])
    df['z_dropoff'] = np.sin(df['dropoff_latitude'])

    df.drop(['pickup_latitude', 'pickup_longitude'], axis = 1, inplace = True)

    return df

def distance_from(loc1,loc2): 
    # Calcola la distanza su due punti della mappa
    dist=hs.haversine(loc1,loc2)
    return round(dist,2)

def compute_distance(df):
    # Crea la coppia di coordinate
    df['coor_pickup'] = list(zip(df['pickup_latitude'], df['pickup_longitude']))
    df['coor_dropoff'] = list(zip(df['dropoff_latitude'], df['dropoff_longitude']))

    # Calcola la distanza
    df['dis'] = df.apply(lambda row: distance_from(row['coor_dropoff'], row['coor_pickup']), axis = 1)
    
    # Camcella le colonne inutili
    df = df.drop(['coor_pickup', 'coor_dropoff'], axis = 1)
    
    return df

def preprocessing_data():
    train, test = open_data()
    print("************* TRAIN DATA ***************\n", train.head())

    print("************* TEST DATA ***************\n",test.head())

    print("************* CHECKING MISSING ON TRAINING ************ \n")
    check_missing(train)
    
    print("************* CHECKING MISSING ON TEST ************ \n")
    check_missing(test)

    print("************* DELETE EXTREME OBSERVATIONS ************ \n")
    # La variabile di target ha un bel po' di outlier quindi leviamoli
    x_train_no_out, y_train_no_out = delete_outliers(train)   

    print("************* SPLITTING DATA ************ \n")        
    x_test = test.copy()

    x_train, x_validation, y_train, y_validation = train_test_split(x_train_no_out, y_train_no_out, test_size=0.3, random_state=4242)
    
    print("************* PRODUCING PLOTS ************ \n")
    # Levo le date per disegnare e levo anche la data di dropoff nel train perché non è presente nel test
    train_for_distribution = x_train.drop(['pickup_datetime', 'dropoff_datetime'],axis =1)
    validation_for_distribution = x_validation.drop(['pickup_datetime', 'dropoff_datetime'],axis =1)
    test_for_distribution = x_test.drop('pickup_datetime', axis =1)
    
    histo_plot(train_for_distribution, validation_for_distribution, test_for_distribution)
    target_distribution(y_train['trip_duration'], y_validation['trip_duration'])

    print("************* COMPUTE DISTANCE BETWEEN POINTS ************ \n")  
    x_train = compute_distance(x_train)
    x_validation = compute_distance(x_validation)
    x_test = compute_distance(x_test)

    print("************* FIX LATITUDE AND LONGITUDE ************ \n")   
    x_train = fix_lat_long(x_train)
    x_validation = fix_lat_long(x_validation)
    x_test = fix_lat_long(x_test)

    print("************* SCALE THE DATA ************ \n")
    # Scaliamo i dati ricordandoci di non scalare le date

    col_names = ['number_of_reviews', 'calculated_host_listings_count', 'Private_room', 'minimum_nights',
             'Entire_home/apt','reviews_per_month','x','y','z','availability_365']

    features_train = X_train[col_names]
    features_validation = X_validation[col_names]
    features_test = X_test[col_names]

    ct = ColumnTransformer([
            ('somename', StandardScaler(), ['number_of_reviews', 'calculated_host_listings_count',
                                            'reviews_per_month','x','y','z','availability_365'])], 
        remainder='passthrough')

    X_train_scaled = ct.fit_transform(features_train)
    X_validation_scaled = ct.fit_transform(features_validation)
    X_test_scaled = ct.fit_transform(features_test)
    sc_y = StandardScaler()

    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1))
    y_validation_scaled = sc_y.fit_transform(y_validation.reshape(-1, 1))

    joblib.dump(sc_y,  "scaler_y.pkl")

    y_train_scaled = y_train_scaled.reshape(-1,1)
    y_validation_scaled = y_validation_scaled.reshape(-1,1)
    print("************* TRANSFoRM AND SAVE INTO NUMPY ARRAY ************ \n")
    # Salviamo i dati in numpy array in modo da poterli riusare
    #np.save('../data/x_train_no_out.npy', x_train)
    #np.save('../data/y_train_no_out.npy', y_train)
    #np.save('../data/x_validation_no_out.npy', x_validation)
    #np.save('../data/y_validation_no_out.npy', y_validation)
    #np.save('../data/x_test.npy', x_test)

    return x_train, y_train, x_validation, y_validation, x_test