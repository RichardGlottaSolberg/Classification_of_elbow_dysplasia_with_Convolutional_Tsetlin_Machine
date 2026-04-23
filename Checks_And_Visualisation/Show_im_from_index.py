import numpy as np
import h5py
import matplotlib.pyplot as plt

                             
patient_idx_input = 6796                   # which patient index to display

with h5py.File(r'path\to\your\dataset.h5', 'r') as f:
    # Search through all folds to find the patient
    found = False
    for fold_key in f.keys():
        if fold_key.startswith('fold_'):
            idx = np.atleast_1d(f[fold_key]['patient_idx'][()])
            image_nr = np.where(idx == patient_idx_input)[0]
            
            if len(image_nr) > 0:
                FOLD = int(fold_key.split('_')[1])
                image_nr = image_nr[0]
                
                images = f[fold_key]['image']
                diagnosis = f[fold_key]['diagnosis']
                target = f[fold_key]['target']
                
                # Load image and remove channel dimension if present
                img = images[image_nr]
                if len(img.shape) == 3 and img.shape[-1] == 1:
                    img = img.squeeze(-1)  # Remove channel dimension (x, x, 1) -> (x, x)
                
                plt.figure(figsize=(10, 10))
                plt.imshow(img, cmap='gray')
                plt.axis("off")
                plt.title(f"Image {image_nr} in fold {FOLD}, Patient Index {idx[image_nr]}, Diagnosis {diagnosis[image_nr]}, Target {target[image_nr]}, Size {img.shape}")
                plt.show()
                found = True
                break
    
    if not found:
        print(f"No images found for patient index {patient_idx_input} in any fold")