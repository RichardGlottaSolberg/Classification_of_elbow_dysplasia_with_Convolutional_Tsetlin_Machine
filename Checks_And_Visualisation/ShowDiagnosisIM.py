
import numpy as np
import h5py
import matplotlib.pyplot as plt

                             
# loop trough a fold and show all images with specific diagnosis, and show image
showdiag = 3    # Change this to the diagnosis you want to visualize (0, 1, 2 or 3)

with h5py.File(r'path\to\your\dataset.h5', 'r') as f:
    images = f[f'fold_0']['image']
    idx = f[f'fold_0']['patient_idx']
    diagnosis = f[f'fold_0']['diagnosis']
    target = f[f'fold_0']['target']
    
    for i in range(len(diagnosis)):
        if diagnosis[i] == showdiag:
            img = images[i]
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)  # Remove channel dimension (x, x, 1) -> (x, x)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img, cmap='gray')
            plt.axis("off")
            plt.title(f"Image {i} in fold 0, Patient Index {idx[i]}, Diagnosis {diagnosis[i]}, Target {target[i]}, Size {img.shape}")
            plt.show()