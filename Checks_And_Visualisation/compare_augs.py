import numpy as np
import h5py
import matplotlib.pyplot as plt

fold_input = 0  # which fold
image_numbers = [0, 1, 2, 3, 4]  # which image numbers to visualize


datasets = [
    'path/to/perfect.h5',           # Perfect dataset
    'path/to/rotated.h5',           # Rotated version of the perfect dataset
    'path/to/mirrored.h5',          # Mirrored version of the perfect dataset
    'path/to/both.h5',              # Both augmentations 
]

# Create figure with horizontal layout (4 rows, 5 columns) - larger size
fig, axes = plt.subplots(4, 5, figsize=(30, 15), dpi=100)
fig.suptitle(f'Fold {fold_input}, Images {image_numbers} med varierende augmentering', fontsize=28, y=0.995)

for dataset_idx, dataset_path in enumerate(datasets):
    with h5py.File(dataset_path, 'r') as f:
        fold_key = f'fold_{fold_input}'
        
        if fold_key in f:
            images = f[fold_key]['image']
            diagnosis = f[fold_key]['diagnosis']
            target = f[fold_key]['target']
            patient_idx = f[fold_key]['patient_idx'][()]
            
            for col, img_num in enumerate(image_numbers):
                img = images[img_num]
                if len(img.shape) == 3 and img.shape[-1] == 1:
                    img = img.squeeze(-1)
                
                ax = axes[dataset_idx, col]
                ax.imshow(img, cmap='gray')
                ax.axis('off')

plt.tight_layout()
plt.savefig('compare_augs_output.png', dpi=150, bbox_inches='tight')  # Save high quality for Word
plt.show()