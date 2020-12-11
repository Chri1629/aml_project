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
import joblib
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


def scale_data(df):
    col_names = ['dis', 'x_pickup', 'y_pickup', 'z_pickup', 'x_dropoff', 'y_dropoff', 'z_dropoff']
    features = df[col_names]
    ct = ColumnTransformer([
            ('somename', StandardScaler(), ['dis', 'x_pickup', 'y_pickup', 'z_pickup', 'x_dropoff', 'y_dropoff', 'z_dropoff'])], 
            remainder='passthrough')

    df[col_names] = ct.fit_transform(features)
    # Vanno concatenate nel modo giusto, cosa che non fa al momento
    return df


def compute_distance(df_train, df_validation, df_test):
    # Crea la coppia di coordinate
    if not os.path.exists('../data/x_train_no_out_dist.csv'):
        df_train['coor_pickup'] = list(zip(df_train['pickup_latitude'], df_train['pickup_longitude']))
        df_train['coor_dropoff'] = list(zip(df_train['dropoff_latitude'], df_train['dropoff_longitude']))

        df_validation['coor_pickup'] = list(zip(df_validation['pickup_latitude'], df_validation['pickup_longitude']))
        df_validation['coor_dropoff'] = list(zip(df_validation['dropoff_latitude'], df_validation['dropoff_longitude']))

        df_test['coor_pickup'] = list(zip(df_test['pickup_latitude'], df_test['pickup_longitude']))
        df_test['coor_dropoff'] = list(zip(df_test['dropoff_latitude'], df_test['dropoff_longitude']))

        # Calcola la distanza
        df_train['dis'] = df_train.apply(lambda row: distance_from(row['coor_dropoff'], row['coor_pickup']), axis = 1)
        df_validation['dis'] = df_validation.apply(lambda row: distance_from(row['coor_dropoff'], row['coor_pickup']), axis = 1)
        df_test['dis'] = df_test.apply(lambda row: distance_from(row['coor_dropoff'], row['coor_pickup']), axis = 1)

        df_train = df_train.drop(['coor_pickup', 'coor_dropoff'], axis = 1)
        df_validation = df_validation.drop(['coor_pickup', 'coor_dropoff'], axis = 1)
        df_test = df_test.drop(['coor_pickup', 'coor_dropoff'], axis = 1)
        # Salvo il df
        df_train.to_csv('../data/x_train_no_out_dist.csv')
        df_validation.to_csv('../data/x_validation_no_out_dist.csv')
        df_test.to_csv('../data/x_test_no_out_dist.csv')
        
    # Se i dati sono già presenti li carico
    df_train = pd.read_csv('../data/x_train_no_out_dist.csv', sep = ",", index_col=0) 
    df_validation = pd.read_csv('../data/x_validation_no_out_dist.csv', sep = ",", index_col=0) 
    df_test = pd.read_csv('../data/x_test_no_out_dist.csv', sep = ",", index_col=0) 

    return df_train, df_validation, df_test


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
    x_train, x_validation, x_test = compute_distance(x_train, x_validation, x_test)

    print("************* FIX LATITUDE AND LONGITUDE ************ \n")   
    x_train = fix_lat_long(x_train)
    x_validation = fix_lat_long(x_validation)
    x_test = fix_lat_long(x_test)

    print("************* CREATE DUMMIES ************ \n")
    x_train = pd.get_dummies(x_train, columns = ['passenger_count','store_and_fwd_flag'])
    x_validation = pd.get_dummies(x_validation, columns = ['passenger_count','store_and_fwd_flag'])
    x_test = pd.get_dummies(x_test, columns = ['passenger_count','store_and_fwd_flag'])

    print("************* SCALE THE DATA ************ \n")
    x_train_scaled = scale_data(x_train)
    x_validation_scaled = scale_data(x_validation)
    x_test_scaled = scale_data(x_test)

    
    # Ora scala la variabile di target
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train)
    y_validation_scaled = sc_y.fit_transform(y_validation)

    # Salva lo scaler di y per poter tornare indietro
    joblib.dump(sc_y,  "scaler_y.pkl")

    y_train_scaled = y_train_scaled.reshape(-1,1)
    y_validation_scaled = y_validation_scaled.reshape(-1,1)

    print("************* DROP UNUSEFUL COLUMNS ************ \n")

    # Droppiamo momentaneamente le colonne che non so come trattare in modo da lavorarci. Poi le utilizzeremo meglio
    # Sarebbe interessante secondo me tenere solo l'ora del timestemp perché penso che sia la più indicativa 

    x_train_scaled = x_train_scaled.iloc[:,6:]
    x_validation_scaled = x_validation_scaled.iloc[:,6:]
    x_test_scaled = x_test_scaled.iloc[:,5:]

    print("************* FIX THE CLASSES IN THE TEST SET ************ \n")
    # Per trattare le cose in questo modo devo aggiungere le colonne fittizie a test per la classe 7 e 8 che non sono presenti in
    # test
    x_validation_scaled.insert(loc=14, column='passenger_count_7', value=0)
    x_validation_scaled.insert(loc=15, column='passenger_count_8', value=0)
    x_validation_scaled.insert(loc=16, column='passenger_count_9', value=0)


    x_test_scaled.insert(loc=14, column='passenger_count_7', value=0)
    x_test_scaled.insert(loc=15, column='passenger_count_8', value=0)

    return x_train_scaled, x_validation_scaled, x_test_scaled, y_train_scaled, y_validation, y_validation_scaled