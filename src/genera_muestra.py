# Para hacer el muestreo por Latin Hypecube
from scipy.stats.qmc import LatinHypercube,scale
import pickle
import numpy as np 

# Tamaño de la población
n = 3000

# Número de variables
n_var = 635

engine = LatinHypercube(d=n_var)
sample = engine.random(n=n)

l_bounds = np.array([0.6]*n_var)
u_bounds = np.array([1.4]*n_var)

sample_scaled = scale(sample,l_bounds, u_bounds)
sample_scaled_dict = {i:j for i,j in enumerate(sample_scaled)}


with open('sample_scaled.pickle', 'wb') as handle:
    pickle.dump(sample_scaled_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

