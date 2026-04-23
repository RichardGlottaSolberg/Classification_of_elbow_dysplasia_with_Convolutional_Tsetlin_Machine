import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Compare original vs altered images side by side
original_file = r'path\to\your\original_dataset.h5'     # Change this to the path of your original dataset
altered_file = r'path\to\your\altered_dataset.h5'       # Change this to the path of your altered dataset

with h5py.File(original_file, 'r') as orig_f, \
     h5py.File(altered_file, 'r') as alt_f:
    
    # Get first few images from fold
    fold = 4
    orig_images = orig_f[f'fold_{fold}']['image'][:5]
    alt_images = alt_f[f'fold_{fold}']['image'][:5]
    
    # Show side by side and display resolution of each
    fig, axes = plt.subplots(5, 2, figsize=(8, 15))
    fig.suptitle('Original vs Altered Images', fontsize=16)
    
    for i in range(5):
        # Handle original image (remove channel dimension if present)
        orig_img = orig_images[i]
        if len(orig_img.shape) == 3:
            orig_img = orig_img[:, :, 0]  # Take first channel if 3D
        
        axes[i, 0].imshow(orig_img, cmap='gray')
        axes[i, 0].set_title(f'Original {i} - {orig_img.shape}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(alt_images[i], cmap='gray')
        axes[i, 1].set_title(f'Altered {i} - {alt_images[i].shape}')
        axes[i, 1].axis('off')
        
        # Resize original to match altered for comparison
        orig_resized = np.array(Image.fromarray(orig_img.astype(np.uint8)).resize(
            alt_images[i].shape[::-1], Image.LANCZOS))
        
        diff = np.sum(np.abs(orig_resized.astype(float) - alt_images[i].astype(float)))
        print(f"Image {i} - Difference sum: {diff:.2f}")
    
    plt.tight_layout()
    plt.show()