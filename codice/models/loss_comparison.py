'''
Programm to compare the loss function for the same network for different activation function 
			Command line in order to compile from shell:
 			python loss_comparison.py -a "risultati_modello"
Christian Uccheddu
'''



#	Import of useful libraries

import numpy as np
import pandas as pd
import matplotlib as mp
import argparse
import matplotlib.pyplot as plt


#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()

parser.add_argument('-c', '--comparison', type=str, required=False, default = "../risultati_modelli", help="Inserire la directory in cui salvare i plot di learning rate")

args = parser.parse_args()


def val_comparison(d1,d2):
    fig = plt.figure(figsize=(10,5)) 
    fig.add_subplot(121)
    plt.plot(d1["loss"], label= "best_model", color = "skyblue")
    plt.plot(d2["loss"], label= "prova", color = "red", alpha = 0.6)
    plt.legend(fontsize = 15)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.xlabel("Epoch", size = 15)
    plt.ylabel("Validation Loss", size = 15)
    plt.title("Validation Loss", size = 15)
    
    fig.add_subplot(122)
    plt.plot(d1["val_loss"], label= "best_model", color = "skyblue")
    plt.plot(d2["val_loss"], label= "prova", color = "red", alpha = 0.6)
    plt.legend(fontsize = 15)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.xlabel("Epoch", size = 15)
    plt.ylabel("Training Loss", size = 15)
    plt.title("Training Loss", size = 15)
    
    plt.show()	
#    fig.savefig('val_loss_activation_comparison.png', bbox_inches='tight', dpi = 100)
    fig.savefig('loss_comparison.png', bbox_inches='tight', dpi = 100)
 
#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<

#   Opening of loss function used with different learning rate

mod_1 = pd.read_csv(args.comparison+"/best_model/training.log")
mod_2 = pd.read_csv(args.comparison+"/prova/training.log")

#   Some useful plots
val_comparison(mod_1,mod_2)
