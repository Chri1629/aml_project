'''
File che definisce i plot esplorativi
'''

import matplotlib.pyplot as plt

def histogram_plot(x,y):

    '''
    Istogramma per disegnare le distribuzioni delle variabili
    @params:
        x:   - Required   : Prima serie di cui disegnare la distribuzione
        y:   - Required   : Seconda serie di cui disegnare la distribuzione
    '''

    fig = plt.figure()
    plt.hist(x, bins = 1000,  color = "skyblue", label = "Training set")
    plt.hist(y, bins = 1000,  color = "red", label= "Test set")
    plt.xlim(min(x), max(x))
    plt.xlabel("Bins di f{}".format(x.name))
    plt.ylabel("Count")
    plt.grid()
    plt.title('Distribuzione di f{}'.format(x.name))
    plt.legend()
    plt.show()
    fig.savefig("preprocessing/pics/Histogram_f{}.png".format(x.name), format = "png", dpi = 200)