# Advancede Machine Learning project

## Preparation

We suggest to install all the useful libraries in a virtual environment in order to have the same version and to not create confilcts and warnings.

First of all in a command line:

`virtualenv env`

Then activate it

`source env/bin/activate`

Then install al the libraries used with:

`pip install -r requirements.txt`

If some of the packages are already installed in the PC this operation is not too slow because it uses the cache memory to install the libraries


## General structure

For the execution we suggest to use the following structure and to create the directory **scaled** inside the directory **data** that store all the data after the preprocessing phase in order to do the preprocessing only one time because it is an expensive computation.

```
project
│   README.md
│   requirements.txt    
│
└───codice
│   └───preprocessing
│   │   │   explorative_plots.py
│   │   │   open.py
│   │   │   preprocessing.py
│   │   │   __init__.py
│   └───models
│   │   │   loss_comparison.py
│   │   │   computation.py
│   │   │   all_models.py
│   │   │   __init__.py
│   └───explorative_pics
│   │   │   map.png
│   │   │   passenger_trips.png
│   │   │   ...
│   └───risultati_modelli
│       └───modello_1
│       └───modello_2
│       └───modello_3
└───data
    └───scaled
    │   train.csv
    │   test.csv
```

## Execution

In order to execute all the files you must run:

`python main.py`

There are some paramaters that can be specified on the command line in order to change the hyperparameters without opening the code. In particular:

```
'-e', '--epoch', type=int, required=False, help ="Give the number of epochs, default = 5"

'-ie', '--initial_epoch', type=int, required=False, default=0, help ="Give the initial epoch, default = 0"

'-bs', '--batch-size', type=int, required=False, default=128, help ="Give the batch size, default = 128"

'-lr', '--learning-rate', type=float, required=False, default=5e-3, help ="Give the learning rate, default = 5e-3"

'-d', '--data', type=str, required=False, default="data", help = "Give the directory of the data"

'-p', '--patience', type=str, required=False, default="0.01:50", help="Patience format:  delta_val:epochs"

'-m', '--model', type=str, required=False

'-ev', '--evaluate', action="store_true"

'-sst', '--save-steps', action="store_true"

'-dr', '--decay-rate', type=float, required=False, default=0, help ="Give the decay rate")
parser.add_argument('-ms', '--model-schema', type=str, required=False, default = "model1", help="Model structure"

'-o', '--output', type=str, required=False, default="risultati_modelli"
```

An example of a correct execution of the code is:

`python main.py -o "risultati_modelli/prova" -ms model1 -e 10 -lr 0.001`
## Additional plots

In order to reproduce all the plots that have been made:

`cd preprocessing`

`python explorative_plots.py`
