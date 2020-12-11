'''
File che definisce i plot esplorativi
'''

import matplotlib.pyplot as plt
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, required=False, default="risultati_modelli")

args = parser.parse_args()
pd.options.mode.chained_assignment = None

def target_distribution(df_train, df_validation):

    '''
    Istogramma per disegnare le distribuzioni delle variabili
    @params:
        x:   - Required   : Prima serie di cui disegnare la distribuzione
        y:   - Required   : Seconda serie di cui disegnare la distribuzione
    '''
    
    fig = plt.figure(figsize = (8,4))
    plt.suptitle("Distribution of target variable")
    plt.subplot(1,2,1)
    plt.hist(df_train, bins = 100)
    plt.title("training")
    plt.subplot(1,2,2)
    plt.hist(df_validation, bins = 100)
    plt.title("validation")
    #plt.show()
    fig.savefig("explorative_pics/target_distribution.png", dpi =100, bbox_inches='tight')
    plt.close(fig)

def acc_loss(history):
    fig = plt.figure(figsize=(15,20))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label = "Train loss", color = "red")
    plt.plot(history.history['val_loss'], label="Validation loss", color = "skyblue")
    plt.legend(loc='upper right', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel('Epochs', size = 15)
    plt.ylabel('Loss', size = 15) 
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label = "Train Accuracy", color = "red")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy", color = "skyblue")
    plt.legend(loc='upper right', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel('Epochs', size = 15)
    plt.ylabel('Accuracy', size = 15)
    plt.show()
    fig.savefig(args.output + "/pics/acc_loss.png", dpi =100, bbox_inches='tight')
    plt.close(fig) 

def histo_plot(df1, df2, df3):
    for i, colonna in enumerate(df1.columns[1:]):
        fig = plt.figure(i,figsize = (8,4))
        plt.suptitle("Distribution of {}".format(colonna))
        plt.subplot(1,3,1)
        plt.hist(df1[colonna], bins = 100)
        plt.title("Training")
        plt.subplot(1,3,2)
        plt.hist(df2[colonna], bins = 100)
        plt.title("Validation")
        plt.subplot(1,3,3)
        plt.hist(df3[colonna], bins = 100)
        plt.title("Test")
        #plt.show()
        fig.savefig("explorative_pics/distribution_{}.png".format(colonna), dpi =100, bbox_inches='tight')
        plt.close(fig)