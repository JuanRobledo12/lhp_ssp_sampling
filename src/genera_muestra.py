'''
Generates a picke file with "n" matrices of shape (n_var,).
This vector will multiply all the input columns element-wise
'''


# Para hacer el muestreo por Latin Hypecube
from scipy.stats.qmc import LatinHypercube,scale
import pickle
import numpy as np 


class GenerateLHCSample:

    def __init__(self, n, n_var):
        
        # Tamaño de la población
        self.n = n
        # Número de variables
        self.n_var = n_var

        self.engine = LatinHypercube(d=n_var)
        self.sample = self.engine.random(n=n)

        self.l_bounds = np.array([0.0]*self.n_var)
        self.u_bounds = np.array([2.0]*self.n_var)

    
    def generate_sample(self):

        sample_scaled = scale(self.sample, self.l_bounds, self.u_bounds)
        sample_scaled_dict = {i:j for i,j in enumerate(sample_scaled)}


        with open('sample_scaled.pickle', 'wb') as handle:
            pickle.dump(sample_scaled_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

