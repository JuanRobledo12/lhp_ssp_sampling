# Para hacer el muestreo por Latin Hypercube
from scipy.stats.qmc import LatinHypercube, scale
import pickle
import numpy as np
import os

class GenerateLHS:

    def __init__(self, n, n_var, u_bound, mode='vector', m=None):
        """
        Initialize the GenerateLHS class.

        Parameters:
        - n: Number of samples (vectors or matrices)
        - n_var: Number of variables
        - u_bound: Upper bound for the sampling
        - mode: 'vector' for n vectors, 'matrix' for n matrices of size m x n_var
        - m: Number of rows for each matrix (only used when mode='matrix')
        """
        self.n = n
        self.n_var = n_var
        self.u_bound_scalar = u_bound
        self.mode = mode
        self.m = m if mode == 'matrix' else None

        # Create the Latin Hypercube engine
        self.engine = LatinHypercube(d=n_var if mode == 'vector' else n_var * m)
        self.sample = self.engine.random(n=n)

        # Bounds for scaling
        self.l_bounds = np.array([0.0] * (n_var if mode == 'vector' else n_var * m))
        self.u_bounds = np.array([u_bound] * (n_var if mode == 'vector' else n_var * m))

    def generate_sample(self):
        """
        Generate Latin Hypercube samples scaled to the specified bounds.
        """
        # Scale the samples
        sample_scaled = scale(self.sample, self.l_bounds, self.u_bounds)

        # Reshape if the mode is 'matrix'
        if self.mode == 'matrix':
            sample_scaled = sample_scaled.reshape(self.n, self.m, self.n_var)

        # Create a dictionary for saving
        sample_scaled_dict = {i: j for i, j in enumerate(sample_scaled)}

        # File path for saving
        sampling_file_path = os.path.join(
            'sampling_files', f'sample_scaled_{self.n}_{self.u_bound_scalar}_{self.mode}.pickle'
        )
        
        # Save the sample to a pickle file
        with open(sampling_file_path, 'wb') as handle:
            pickle.dump(sample_scaled_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

