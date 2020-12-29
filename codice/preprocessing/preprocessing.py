from .open import open_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy import stats
import pandas as pd
import haversine as hs
import joblib
import os
import urllib.request, json 
# geopandas for county
import geopandas
import geopandas.tools
from shapely.geometry import Point

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
        target_no_out = df_train_target[(df_train_target < df_train_target.quantile(.95)) & (df_train_target > 10)]
        
        # Aggiungo nuovamente le altre colonne
        train_no_out = pd.merge(target_no_out, df_train_input, how = "inner", left_index=True, right_index=True)
        
        x_train_no_out = train_no_out.copy()

        # Salva in numpy array
        x_train_no_out.to_csv('../data/x_train_no_out.csv', index = False)
        
    # Se i dati sono già presenti li carico
    x_train_no_out = pd.read_csv('../data/x_train_no_out.csv', sep = ",")

    return x_train_no_out

def fix_lat_long(df):
    df['x_pickup'] = np.cos(df['pickup_latitude']) * np.cos(df['pickup_longitude'])
    df['y_pickup'] = np.cos(df['pickup_latitude']) * np.sin(df['pickup_longitude'])
    df['z_pickup'] = np.sin(df['pickup_latitude'])

    df['x_dropoff'] = np.cos(df['dropoff_latitude']) * np.cos(df['dropoff_longitude'])
    df['y_dropoff'] = np.cos(df['dropoff_latitude']) * np.sin(df['dropoff_longitude'])
    df['z_dropoff'] = np.sin(df['dropoff_latitude'])

    df.drop(['pickup_latitude', 'pickup_longitude'], axis = 1, inplace = True)

    return df


def add_date_info(x_df):
    # aggiunge informazioni alla data
    # date, hour, dayweek
    x_df['pickup_date'] = pd.to_datetime(x_df['pickup_datetime']).dt.date
    x_df['pickup_hour'] = pd.to_datetime(x_df['pickup_datetime']).dt.hour
    x_df['pickup_weekday'] = pd.to_datetime(x_df['pickup_date']).dt.weekday

    return x_df

def add_county(df):
    df["pick_geometry"] = df.apply(lambda row: Point(row["pickup_longitude"], row["pickup_latitude"]), axis=1)
    df = geopandas.GeoDataFrame(df, geometry="pick_geometry")

    map_data = geopandas.GeoDataFrame.from_file("../data/map_files/ZillowNeighborhoods-NY.shp")
    # Drop tutte le colonne tranne conty e geometry
    map_data = map_data[["County", "geometry"]]
    # faccio lo spatial join con il poligono
    gdf = geopandas.tools.sjoin(df, map_data, how="left")
    gdf.drop_duplicates(subset = ["id"], inplace = True)
    # add external category
    gdf["County"].fillna(value = "External", inplace = True)
    # rimuovo colonne inutili
    df_out = pd.DataFrame(gdf.drop(columns = ["pick_geometry", "index_right"]))

    return df_out

def distance_from(loc1,loc2): 
    # Calcola la distanza su due punti della mappa
    dist=hs.haversine(loc1,loc2)
    return round(dist,2)


def distance_from2(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    with urllib.request.urlopen("http://localhost:5000/route/v1/driving/{},{};{},{}?overview=false".format(pickup_long, pickup_lat, dropoff_long, dropoff_lat)) as url: 
        data = json.loads(url.read().decode())
        distance = data["routes"][0]["legs"][0]["distance"]
    return distance


def get_distance(lng1_r=None, lat1_r=None, 
                            lng2_r=None, lat2_r=None  ):
        lat = lat2_r - lat1_r
        lng = lng2_r - lng1_r
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(lng * 0.5) ** 2
        h = 2 * 6371 * np.arcsin(np.sqrt(d))
        return h


def get_distance_manhattan(lat1, lng1, lat2, lng2):
    lat1_r, lng1_r, lat2_r, lng2_r = map( np.radians, (lat1, lng1, lat2, lng2))
    a = get_distance(lng1_r, lat1_r, lng2_r, lat1_r )
    b = get_distance(lng1_r, lat1_r, lng1_r, lat2_r )
    return a + b 
    

def scale_data(df):

    ct = ColumnTransformer([
            ('somename', StandardScaler(), ['dis', 'x_pickup', 'y_pickup', 'z_pickup', 'x_dropoff', 'y_dropoff', 'z_dropoff']),
            ('categorical', OneHotEncoder(), ['pickup_hour', 'pickup_weekday'])],
            remainder='passthrough')

    df = ct.fit_transform(df)

    return df


def compute_distance(df_train, df_test):
    # Crea la coppia di coordinate
    if not os.path.exists('../data/x_train_no_out_dist.csv'):

        # aggiungo dati ora e giorno settimana
        df_train = add_date_info(df_train)
        df_test = add_date_info(df_test)

        df_train['coor_pickup'] = list(zip(df_train['pickup_latitude'], df_train['pickup_longitude']))
        df_train['coor_dropoff'] = list(zip(df_train['dropoff_latitude'], df_train['dropoff_longitude']))

        df_test['coor_pickup'] = list(zip(df_test['pickup_latitude'], df_test['pickup_longitude']))
        df_test['coor_dropoff'] = list(zip(df_test['dropoff_latitude'], df_test['dropoff_longitude']))

        df_train['dis'] = df_train.apply(lambda row: get_distance_manhattan(row["pickup_longitude"],row["pickup_latitude"],row["dropoff_longitude"],row["dropoff_latitude"]), axis = 1)
        df_test['dis'] = df_test.apply(lambda row: get_distance_manhattan(row["pickup_longitude"],row["pickup_latitude"],row["dropoff_longitude"],row["dropoff_latitude"]), axis = 1)

        df_train = df_train.drop(['coor_pickup', 'coor_dropoff'], axis = 1)
        df_test = df_test.drop(['coor_pickup', 'coor_dropoff'], axis = 1)
        # Salvo il df
        df_train.to_csv('../data/x_train_no_out_dist.csv', index = False)
        df_test.to_csv('../data/x_test_no_out_dist.csv', index = False)
        
    # Se i dati sono già presenti li carico
    df_train = pd.read_csv('../data/x_train_no_out_dist.csv', sep = ",") 
    df_test = pd.read_csv('../data/x_test_no_out_dist.csv', sep = ",") 

    return df_train, df_test


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
    x_train_no_out = delete_outliers(train)   

    print("************** ADD COUNTY INFORMATION **************** \n")
    if not os.path.exists('../data/x_train_no_out_dist.csv'):
        x_train_no_out = add_county(x_train_no_out)
        x_test = add_county(test)
        print("************* CREATE DUMMIES ************ \n")
        x_train_no_out = pd.get_dummies(x_train_no_out, columns = ['passenger_count','store_and_fwd_flag', "County"])
        x_test = pd.get_dummies(x_test, columns = ['passenger_count','store_and_fwd_flag', "County"])
        print("************* COMPUTE DISTANCE BETWEEN POINTS + DATE-HOUR ************ \n")  
        x_train, x_test = compute_distance(x_train_no_out, x_test)
    else:
        x_train = pd.read_csv('../data/x_train_no_out_dist.csv')
        x_test = pd.read_csv('../data/x_test_no_out_dist.csv')

    print('outliers in distance in df_train:', len(x_train[x_train['dis'] > 100]))
    x_train = x_train[x_train['dis'] < 100]

    print("************* FIX LATITUDE AND LONGITUDE ************ \n")   
    x_train = fix_lat_long(x_train)
    x_test = fix_lat_long(x_test)
    
    print("************* FIX THE CLASSES IN THE TEST SET ************ \n")
    x_test.insert(loc=12, column='passenger_count_7', value=0)
    x_test.insert(loc=13, column='passenger_count_8', value=0)
    x_test.insert(loc=25, column='County_Suffolk', value=0)
    


    print("************* DROP UNUSEFUL COLUMNS ************ \n")

    # Droppiamo momentaneamente le colonne che non so come trattare in modo da lavorarci. Poi le utilizzeremo meglio
    # Sarebbe interessante secondo me tenere solo l'ora del timestemp perché penso che sia la più indicativa 
    y_train = x_train.iloc[:,0]
    x_train = x_train.iloc[:,7:].drop(['pickup_date'], axis = 1)
    x_test = x_test.iloc[:,5:].drop(['pickup_date'], axis = 1)
    
    print("************* SPLITTING DATA ************ \n")        
    
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.3, random_state=4242)

    print("************* SCALE THE DATA ************ \n")
    
    x_train_scaled = scale_data(x_train)
    x_validation_scaled = scale_data(x_validation)
    x_test_scaled = scale_data(x_test)

    #Ora scala la variabile di target
    sc_y = StandardScaler()

    print("pre reshape",y_train.shape, y_validation.shape)
    y_train = y_train.to_numpy()
    y_train = y_train.reshape(-1,1)

    y_validation = y_validation.to_numpy()
    y_validation = y_validation.reshape(-1,1)
    print("after reshape",y_train.shape, y_validation.shape)
    
    y_train_scaled = sc_y.fit_transform(y_train) 
    y_validation_scaled = sc_y.fit_transform(y_validation)

    # Salva lo scaler di y per poter tornare indietro
    joblib.dump(sc_y,  "scaler_y.pkl")
    
    return x_train_scaled, x_validation_scaled, x_test_scaled, y_train_scaled, y_validation, y_validation_scaled