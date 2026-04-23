import numpy as np
import h5py
import matplotlib.pyplot as plt

                             
FOLD = 0                                # which fold to visualize
image_nr = 0                            # which image to visualize

with h5py.File(r'path\to\your\dataset.h5', 'r') as f:
    images = f[f'fold_{FOLD}']['image']
    idx = f[f'fold_{FOLD}']['patient_idx']
    diagnosis = f[f'fold_{FOLD}']['diagnosis']
    target = f[f'fold_{FOLD}']['target']
    
    # Load image and remove channel dimension if present
    img = images[image_nr]
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)  # Remove channel dimension (x, x, 1) -> (x, x)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title(f"Image {image_nr} in fold {FOLD}, Patient Index {idx[image_nr]}, Diagnosis {diagnosis[image_nr]}, Target {target[image_nr]}, Size {img.shape}")
    plt.show()