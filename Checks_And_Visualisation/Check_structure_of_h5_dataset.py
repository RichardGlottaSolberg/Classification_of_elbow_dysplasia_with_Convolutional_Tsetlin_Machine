# imports
import h5py
import numpy as np

# Display structure of h5 file

file = r'path\to\your\dataset.h5'  # Change this to the path of the h5 dataset to explore

def print_structure(file):
    
    with h5py.File(file, 'r') as f:
        f.visititems(print)
        
print_structure(file)