'''
Main module to execute all the steps of the project with the following command line parameters:

    - e (--epoch): Give the number of epochs, default = 5
    - ie (--initial_epoch): Give the initial epoch, default = 0
    - bs (--batch-size): Give the batch size, default = 128
    - lr (--learning-rate): Give the learning rate, default = 5e-3
    - d (--data): Give the directory of the data
    - p (--patience): Patience format:  delta_val:epochs
    - m (--model): 
    - ev (--evaluate):
    - sst (--save-steps):
    - ms (--model-schema): Model structure, default = model1
    - o (--output): Output directory path, default = risultati_modelli
'''
from preprocessing.preprocessing import preprocessing_data

from models import computation
import time
import argparse
import pandas as pd



pd.options.mode.chained_assignment = None


def mkdir_p(mypath):
    '''
    Create directories that are used if they not exist
    Parameters:
        mypath  -Required:  output path chosen
    '''
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def main():
    '''
    Main function that execute computation
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, required=False, default=5, help ="Give the number of epochs, default = 5")
    parser.add_argument('-ie', '--initial_epoch', type=int, required=False, default=0, help ="Give the initial epoch, default = 0")
    parser.add_argument('-bs', '--batch-size', type=int, required=False, default=128, help ="Give the batch size, default = 128")
    parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=5e-3, help ="Give the learning rate, default = 5e-3") 
    parser.add_argument('-d', '--data', type=str, required=False, default="data", help = "Give the directory of the data")
    parser.add_argument('-p', '--patience', type=str, required=False, default="0.01:50",
                        help="Patience format:  delta_val:epochs")
    parser.add_argument('-m', '--model', type=str, required=False)
    parser.add_argument('-ev', '--evaluate', action="store_true")
    parser.add_argument('-sst', '--save-steps', action="store_true")
    parser.add_argument('-ms', '--model-schema', type=str, required=False, default = "model1", help="Model structure")
    parser.add_argument('-o', '--output', type=str, required=False, default="risultati_modelli")

    args = parser.parse_args()
    mkdir_p(args.output)
    mkdir_p(args.output + "/pics")
    computation(args)
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time \n--- %s s ---" % (time.time() - start_time))