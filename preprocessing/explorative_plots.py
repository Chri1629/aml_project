'''
File che definisce i plot esplorativi
'''

import matplotlib.pyplot as plt

def histogram_plot(x):

    '''
    Istogramma per disegnare le distribuzioni delle variabili
    @params:
        x:   - Required   : Serie di cui disegnare la distribuzione
    @return:
    	l: lista di directory
    '''

    plt.figure()
    plt.hist(x, bins = 1000,  color = "skyblue")
    plt.xlim(min(x), max(x))
    plt.grid()
    plt.title('Distribuzione di f{} '.format(x.name))
    plt.show()
