'''
Program to compare the loss function for different networks 
			Command line in order to compile from shell:
 			python loss_comparison.py -a "risultati_modello"
'''


#	Import of useful libraries

import numpy as np
import pandas as pd
import matplotlib as mp
import argparse
import matplotlib.pyplot as plt


#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments


def val_comparison(d1,d2,d3):
    '''
    Graphs that compares the losses of three different models
    Parameters:
        d1:   - Required   : First neural network
        d2:   - Required   : Second neural network
        d3:   - Required   : Second neural network
    '''

    fig = plt.figure(figsize=(20,10)) 
    fig.add_subplot(121)
    plt.plot(d1["loss"], label= "Model 1", color = "skyblue")
    plt.plot(d2["loss"], label= "Model 2", color = "red", alpha = 0.6)
    plt.plot(d3["loss"], label= "Model 3", color = "green", alpha = 0.6)
    plt.legend(fontsize = 15)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.xlabel("Epoch", size = 15)
    plt.ylabel("Validation Loss", size = 15)
    plt.title("\n Validation Loss \n", size = 15)
    
    fig.add_subplot(122)
    plt.plot(d1["val_loss"], label= "Model 1", color = "skyblue")
    plt.plot(d2["val_loss"], label= "Model 2", color = "red", alpha = 0.6)
    plt.plot(d3["val_loss"], label= "Model 3", color = "green", alpha = 0.6)
    plt.legend(fontsize = 15)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.xlabel("Epoch", size = 15)
    plt.ylabel("Training Loss", size = 15)
    plt.title("\n Training Loss \n", size = 15)
    plt.show()	
    fig.savefig('loss_comparison.png', bbox_inches='tight', dpi = 100)
 
#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--comparison', type=str, required=False, default = "../risultati_modelli", help="Inserire la directory in cui salvare i plot di learning rate")
    args = parser.parse_args()

    #   Opening of loss function used with different learning rate
    mod_1 = pd.read_csv(args.comparison+"/fede4-3/training.log")
    mod_2 = pd.read_csv(args.comparison+"/fede5-2/training.log")
    mod_3 = pd.read_csv(args.comparison+"/fede5-3/training.log")

    #   Some useful plots
    val_comparison(mod_1,mod_2,mod_3)