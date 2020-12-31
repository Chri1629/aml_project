'''
File che definisce i plot esplorativi
'''

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
import seaborn

pd.options.mode.chained_assignment = None

def target_distribution(df_train, df_validation):
    '''
    Histogram to visualize variables distributions 
    Parameters:
        df_train:        - Required   :  train target variable
        df_validation:   - Required   :  val target variable
    '''
    fig = plt.figure(figsize = (15,8))
    plt.hist(df_train, bins = 300, color = "skyblue", alpha = 0.7, label = "Train")
    plt.hist(df_validation, bins = 300, color = "orange", alpha = 0.7, label = "Validation")
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.xlabel('Value', size = 10)
    plt.ylabel('Count', size = 10)
    plt.grid()
    plt.legend(fontsize = 12)
    plt.title("Distribution of target variable", size = 15)
    fig.savefig("../explorative_pics/target_distribution.png", dpi =100, bbox_inches='tight')
    plt.close(fig)

 
def loss_plotter(history, output_path):
    '''
    Plots the loss of the model 
    Parameters:
        history:    - Required   : History of the model fit
        output_path  - Required   : Path in which to save the image
    '''
    fig = plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label = "Train loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.legend(loc='upper right', fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Epochs', size = 20)
    plt.ylabel('Value', size = 20)
    plt.title("Loss function", size =25)
    plt.grid()
    plt.show()
    fig.savefig(output_path + '/pics/loss_plot.png', bbox_inches='tight')
    plt.close(fig)

def scatter_plotter(y, pred, output_path):
    '''
    Scatter plot of true vs predicted trip duration
    Parameters:
        y:    - Required   : True trip duration
        pred:    - Required   : Predicted trip duration
        output_path  - Required   : Path in which to save the image
    '''
    fig = plt.figure(figsize=(15,8))
    plt.plot(y,pred, 'o',markersize=1, label = "Predicted")
    top = max(pred)
    plt.plot([0,top],[0,top], "r--", color = "deepskyblue", label = "Perfect prediction")
    plt.xticks( fontsize = 15)
    plt.yticks( fontsize = 15)
    plt.xlabel('True trip duration', size = 18) 
    plt.ylabel('Predicted trip duration', size = 18) 
    plt.title("Scatter plot of the prediction \n", size = 22)
    plt.legend(loc="best", prop={'size': 15})
    plt.xlim(left = 0)
    plt.ylim(bottom=0)
    plt.ioff()
    plt.grid()
    plt.show()
    fig.savefig(output_path + '/pics/scatter_plot.png', bbox_inches='tight')
    plt.close(fig)

def histo_plot(df1, df2, df3):
    '''
    Istogramma per disegnare le distribuzioni delle variabili
    Parameters:
        df_1:   - Required   : Dataframe di train di cui disegnare la distribuzione
        df_2:   - Required   : Dataframe di validation di cui disegnare la distribuzione
        df_3:   - Required   : Dataframe di test di cui disegnare la distribuzione
    '''
    for i, colonna in enumerate(df1.columns[1:]):
        fig = plt.figure(i,figsize = (16,6))
        plt.suptitle("Distribution of {}".format(colonna), size = 15)
        plt.subplot(1,3,1)
        plt.hist(df1[colonna], bins = 100, color = "skyblue", alpha = 0.7)
        plt.title("Training", size = 12)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.xlabel('Value', size = 10)
        plt.ylabel('Count', size = 10)
        plt.grid()
        plt.subplot(1,3,2)
        plt.hist(df2[colonna], bins = 100, color = "orange", alpha = 0.7)
        plt.title("Validation", size = 12)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.xlabel('Value', size = 10)
        plt.ylabel('Count', size = 10)
        plt.grid()
        plt.subplot(1,3,3)
        plt.hist(df3[colonna], bins = 100, color = "red", alpha = 0.7)
        plt.title("Test", size = 12)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.xlabel('Value', size = 10)
        plt.ylabel('Count', size = 10)
        plt.grid()
        fig.savefig("../explorative_pics/distribution_{}.png".format(colonna), dpi =100, bbox_inches='tight')
        plt.close(fig)


def weekday_trips(df):
    '''
    Visualizzazione della quantità di trips divise per giorno della settimana
    '''
    nomalize=1000 
    fig, ax = plt.subplots(ncols=1, sharey=False)
    fig.set_size_inches(20,6)
    ax.plot(df.groupby('pickup_weekday').count()['vendor_id']/nomalize,  'r-*',  markersize=15,  label='Pickup')
    ax.tick_params(labelsize=13)
    ax.set_xlabel("weekday", fontsize=18)
    ax.set_ylabel('Trips (x%d)'%nomalize, fontsize=18)
    fig.savefig("../explorative_pics/weekday_trips.png", dpi =100, bbox_inches='tight')
    plt.close(fig)


def make_hist( X, xmax=None, xmin=None, binw=1, xlabel='Input', ylabel='Counts', 
              xunit='', edgecolor='black', tightLabel=False, centerLabel=False, debug=False, log=False, **hist_kwds ): 
       
    if not xmax:
        xmax = max(X)+binw
    if not xmin:
        xmin = min(X) if (min(X) < 0) or (min(X) >=1) else 0 
        
    if xmax <= xmin: 
        xmax = max(X)+binw
        xmin = min(X) if (min(X) < 0) or (min(X) >=1) else 0 

    hist_info_ = plt.hist( 
                           x = X[ (X <= xmax) & (X >= xmin) ],
                           bins = np.arange(xmin, xmax+binw, binw), # Due to end 2 bins are 1, hist will combined them to a bin. 
                           edgecolor = edgecolor,
                           log=log,
                           **hist_kwds
                         )
    
    if xunit == '': 
        ylabel = ylabel+' / %.2f'%(binw)
    else:
        xlabel = xlabel+' [%s]'%(xunit)
        ylabel = ylabel+' / %.2f %s'%(binw, xunit)
    plt.tick_params(labelsize=20)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.ylim(ymin = 0.5 if log else 0)
    # Show label be bin by bin
    if tightLabel: 
        plt.xticks(hist_info_[1])
    # Make label be in bins' center
    if tightLabel and centerLabel:
        ax_min = min(hist_info_[1])
        ax_max = max(hist_info_[1])
        ax_wth = (hist_info_[1][1]-hist_info_[1][0])/2.
        newrange = np.arange(ax_min, ax_max+ax_wth, ax_wth)
        newnames, n = [], 0
        for i in range(len(newrange)):
            if i%2 == 0: 
                newnames.append('')
            elif n < len(hist_info_[1]): 
                v = hist_info_[1][n] if hist_info_[1][n]%1 != 0 else int(hist_info_[1][n])
                newnames.append(v)
                n+=1   
        plt.xticks(newrange, newnames)
        if debug: print(ax_min, ax_max, ax_wth, newrange)
    return hist_info_


def pickup_dropoff(df):
    df['pickup_time'] = pd.to_datetime(df['pickup_datetime']).dt.hour + pd.to_datetime(df['pickup_datetime']).dt.minute/60 + pd.to_datetime(df['pickup_datetime']).dt.second/3600
    df['dropoff_time'] = pd.to_datetime(df['dropoff_datetime']).dt.hour + pd.to_datetime(df['dropoff_datetime']).dt.minute/60 + pd.to_datetime(df['dropoff_datetime']).dt.second/3600
    fig = plt.figure(figsize=(20,6))
    pick_info = make_hist(df['pickup_time'].values,  xmax=25, histtype='step', edgecolor='r', linewidth=2, xlabel="Day time (o'clock)", ylabel="Trips", label='Pickup',  tightLabel=True)
    drop_info = make_hist(df['dropoff_time'].values, xmax=25, histtype='step', edgecolor='b', linewidth=2, xlabel="Day time (o'clock)", ylabel="Trips", label='Dropoff', tightLabel=True)
    plt.legend(loc='upper left', fontsize=25)
    fig.savefig("../explorative_pics/pickup_dropoff.png", dpi =100, bbox_inches='tight')
    plt.close()

def passenger_trips(df):
    fig = plt.figure(figsize=(20,6))
    hist_info = make_hist(df['passenger_count'].values, xlabel='Passengers', ylabel='Trips', log=True, tightLabel=True, centerLabel=True)
    fig.savefig("../explorative_pics/passenger_trips.png", dpi =100, bbox_inches='tight')
    plt.close()

def maps(df):
    '''
    Plots the map of the pickup locations in NYC, divided by County
    Parameters:
        df:    - Required   : Dataframe from which to print the map
    '''
    fig = plt.figure(figsize=(12,10))
    s_marker = 5
    alpha_dot = 1
    plt.scatter( x= df[df["County"] == "New York"]["pickup_longitude"], 
                y= df[df["County"] == "New York"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "firebrick", label = "New York")
    plt.scatter( x= df[df["County"] == "Kings"]["pickup_longitude"], 
                y= df[df["County"] == "Kings"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "steelblue", label = "Kings")
    plt.scatter( x= df[df["County"] == "Queens"]["pickup_longitude"], 
                y= df[df["County"] == "Queens"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "forestgreen", label = "Queens")
    plt.scatter( x= df[df["County"] == "External"]["pickup_longitude"], 
                y= df[df["County"] == "External"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "dimgray", label = "External")
    plt.scatter( x= df[df["County"] == "Bronx"]["pickup_longitude"], 
                y= df[df["County"] == "Bronx"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "teal", label = "Bronx")
    plt.scatter( x= df[df["County"] == "Richmond"]["pickup_longitude"], 
                y= df[df["County"] == "Richmond"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "darkviolet", label = "Richmond")
    plt.scatter( x= df[df["County"] == "Westchester"]["pickup_longitude"], 
                y= df[df["County"] == "Westchester"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "olive", label = "Westchester")
    plt.scatter( x= df[df["County"] == "Suffolk"]["pickup_longitude"], 
                y= df[df["County"] == "Suffolk"]["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "darkorange", label = "Suffolk")
    plt.legend(loc = "upper right", fontsize = 10)
    plt.xlim(-74.1, -73.7)
    plt.ylim(40.57, 40.93)
    plt.tick_params(labelsize=10)
    #plt.title(name, fontsize=18 )
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude',  fontsize=12)
    fig.savefig("../explorative_pics/map.png", dpi =100, bbox_inches='tight')
    plt.close()


def maps_black(df):
    '''
    Plots the map of the pickup locations in NYC
    Parameters:
        df:    - Required   : Dataframe from which to print the map
    '''
    fig = plt.figure(figsize=(12,10))
    s_marker = 5
    alpha_dot = 1
    plt.scatter( x= df["pickup_longitude"],
    y= df["pickup_latitude"], s=s_marker, alpha=alpha_dot, color = "black", label = "New York")
    plt.xlim(-74.1, -73.7)
    plt.ylim(40.57, 40.93)
    plt.tick_params(labelsize=10)
    #plt.title(name, fontsize=18 )
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    fig.savefig("../explorative_pics/map_bw.png", dpi =100, bbox_inches='tight')
    plt.close()


def mse_classes(y_val, y_val_pred, modello):
    '''
    Views mse, mae, rmse and perc divided by trip duration bins
    Parameters:
        y_val:    - Required   : file that contains the true trip duration
        pred:    - Required   : file that contains the predicted trip duration
        output_path  - Required   : neural network model
    '''
    y_val = pd.DataFrame(y_val)
    y_val_pred = pd.DataFrame(y_val_pred)
    y_val['categoria'] = np.where(y_val[0] <= 100, '< 100',
                                np.where((y_val[0] > 100) & (y_val[0] <= 200), '100 - 200', 
                                        np.where((y_val[0] > 200) & (y_val[0] <= 500),  '200 - 500',
                                                np.where((y_val[0] > 500) & (y_val[0] <= 1000), '500 - 1000', ' > 1000'))))
    y_val.columns = ['duration', 'categoria']
    totale = pd.concat([y_val, y_val_pred], axis = 1)
    totale.columns = ['duration', 'categoria', 'predicted']
    totale['mse'] = (totale['duration'] - totale['predicted'])**2
    totale['mae'] = abs(totale['duration'] - totale['predicted'])
    totale['perc'] = (abs(totale['duration'] - totale['predicted'])/totale['duration'])*100
    mse_df = pd.DataFrame(columns = ['category', 'mse', 'mae','rmse','perc'])
    for categoria in totale['categoria'].unique():
        a = totale.loc[totale['categoria'] == categoria]
        mse = a['mse'].mean()
        mae = a['mae'].mean()
        rmse = (a['mse'].mean())**(1/2)
        perc = a['perc'].mean()
        mse_df = mse_df.append({'category': categoria,
                            'mse': mse,
                            'mae': mae,
                            'rmse': rmse,
                            'perc': perc}, ignore_index=True)
    mse_df['order'] = np.where(mse_df['category'] == '< 100', 1,
                            np.where(mse_df['category'] == '100 - 200', 2,
                                    np.where(mse_df['category'] == '200 - 500', 3,
                                            np.where(mse_df['category'] == '500 - 1000', 4,5))))
    mse_df = mse_df.sort_values('order')
    mycolors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    fig = plt.figure(figsize = (16,16))
    plt.subplot(2,2,1)
    plt.bar(mse_df['category'], mse_df['mse'], color = mycolors)
    plt.title("\n MSE for category \n", size=24)
    plt.xlabel("Category", size = 18)
    plt.ylabel("MSE", size = 18)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.subplot(2,2,2)
    plt.bar(mse_df['category'], mse_df['mae'], color = mycolors)
    plt.title("\n MAE for category \n", size=24)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.xlabel("Category", size = 18)
    plt.ylabel("MAE", size = 18)
    plt.subplot(2,2,3)
    plt.bar(mse_df['category'], mse_df['rmse'], color = mycolors)
    plt.title("\n RMSE for category \n", size=24)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.xlabel("Category", size = 18)
    plt.ylabel("RMSE", size = 18)
    plt.subplot(2,2,4)
    plt.bar(mse_df['category'], mse_df['perc'], color = mycolors)
    plt.title("\n Perc for category \n", size=24)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.xlabel("Category", size = 18)
    plt.ylabel("Perc", size = 18)
    fig.tight_layout()
    fig.savefig("../risultati_modelli/{}/mse_classes.png".format(modello), dpi =100, bbox_inches='tight')
    plt.close()


# READING FILES

#modello = "fede5-2"
#train_df = pd.read_csv("../../data/x_train_no_out_dist.csv")
#original_train_df =  pd.read_csv("../../data/train.csv")
#y_train = pd.read_csv("../../data/y_train_no_out.csv")
#y_validation = np.load("../../data/scaled/y_validation.npy")
#y_val = np.load("../../data/scaled/y_validation.npy")
#y_val_pred = np.load("../risultati_modelli/{}/y_val_pred.npy".format(modello))

# PLOTTING

#target_distribution(y_train['trip_duration'], y_validation)
#passenger_trips(original_train_df)
#maps(train_df)
#maps_black(train_df)
#pickup_dropoff(train_df)
#weekday_trips(train_df)
#mse_classes(y_val, y_val_pred, modello)