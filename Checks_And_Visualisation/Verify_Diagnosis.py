# Imports
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load original and censored files
original_file = r'path\to\your\original_dataset.h5'  # Change this to the path of your original dataset
altered_file = r'path\to\your\altered_dataset.h5'  # Change this to the path of your altered dataset

# Open both files and verify metadata and diagnosis labels
with h5py.File(original_file, 'r') as orig_f, \
     h5py.File(altered_file, 'r') as alt_f:
    
    # Verify metadata for each fold
    for fold in orig_f.keys():
        orig_fold = orig_f[fold]
        alt_fold = alt_f[fold]
        
        # Check number of images
        orig_num_images = orig_fold['image'].shape[0]
        alt_num_images = alt_fold['image'].shape[0]
        print(f"{fold} - Original images: {orig_num_images}, Altered images: {alt_num_images}")
        
        # Check diagnosis labels
        orig_labels = orig_fold['diagnosis'][:]
        alt_labels = alt_fold['diagnosis'][:]
        if np.array_equal(orig_labels, alt_labels):
            print(f"{fold} - Diagnosis labels match.")
        else:
            print(f"{fold} - Diagnosis labels do NOT match!")
            