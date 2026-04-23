import cv2
import numpy as np
import h5py
from pathlib import Path

INPUT_HDF5 = r"path\to\your\input_dataset.h5"       # Change this to the path of your input dataset of quadraticly cropped images
OUTPUT_HDF5 = r"path\to\your\output_dataset.h5"     # Change this to desired output path for the circularly cropped dataset

with h5py.File(INPUT_HDF5, 'r') as f_in, h5py.File(OUTPUT_HDF5, 'w') as f_out:
    for fold_name in f_in.keys():
        group_in = f_in[fold_name]
        group_out = f_out.create_group(fold_name)
        
        # Copy labels and metadata
        for key in group_in.keys():
            if key != 'image':
                group_out.create_dataset(key, data=group_in[key][()])
        
        # Process images
        images = group_in['image']
        processed_images = []
        
        for img in images:
            # Remove channel dimension if present
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Create circular mask (radius 300 for 600x600 image)
            mask = np.zeros(img.shape, dtype="uint8")
            center = (img.shape[1] // 2, img.shape[0] // 2)  # (300, 300)
            radius = 300
            cv2.circle(mask, center, radius, 255, -1)
            
            # Apply mask to black out pixels outside circle
            circular_img = cv2.bitwise_and(img, img, mask=mask)
            
            processed_images.append(circular_img)
            
        print(f'Processed images in fold {fold_name} with shape {circular_img.shape}')
        # Save processed images to new HDF5
        group_out.create_dataset('image', data=np.array(processed_images), compression="gzip")

print("Circular cropping complete!")
        
            
