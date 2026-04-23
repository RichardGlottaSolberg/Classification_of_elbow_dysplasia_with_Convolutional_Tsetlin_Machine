# Imports
import h5py
import numpy as np
from PIL import Image

# Load the desired 800x800 h5 file for downsizing
input_file = r'path\to\your\input_dataset.h5'       # Change this to the path of dataset you want to resize
output_file = r'path\to\your\output_dataset.h5'     # Change this to the path of dataset you want to create
resize_original_shape = (128, 128)                  # Original shape of the images in the output
downsizeratio = 600/800                             # Ratio to downsize from 800 to 600 (or any other desired size) - adjust as needed
target_size = (int(resize_original_shape[0] * downsizeratio), int(resize_original_shape[1] * downsizeratio))  # New size to resize images to
    
with h5py.File(input_file, 'r') as in_f, \
     h5py.File(output_file, 'w') as out_f:
    
    # Iterate through each fold
    for fold_name in in_f.keys():
        print(f"\nProcessing {fold_name}...")
        fold_group = in_f[fold_name]
        
        # Create corresponding fold group in output file
        out_fold = out_f.create_group(fold_name)
        
        # Copy fold-level attributes if any
        for attr_name, attr_value in fold_group.attrs.items():
            out_fold.attrs[attr_name] = attr_value
        
        # Process each dataset in the fold
        for dataset_name in fold_group.keys():
            dataset = fold_group[dataset_name]
            
            if dataset_name == 'image':
                # Resize images
                print(f"  Resizing images: {dataset.shape}")
                num_images = dataset.shape[0]
                
                # Create output dataset for resized images (grayscale)
                resized_shape = (num_images, target_size[0], target_size[1])
                out_dataset = out_fold.create_dataset(
                    'image',
                    shape=resized_shape,
                    dtype=dataset.dtype,
                    compression='gzip'
                )
                
                # Resize each image
                for i in range(num_images):
                    img_array = dataset[i]
                    img = Image.fromarray(img_array.astype(np.uint8), mode='L')
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    out_dataset[i] = np.array(img_resized)
                    
                    if (i + 1) % 100 == 0:
                        print(f"    Processed {i + 1}/{num_images} images")
                
                # Copy attributes
                for attr_name, attr_value in dataset.attrs.items():
                    out_dataset.attrs[attr_name] = attr_value
            
            else:
                # Copy diagnosis, patient_idx, and target datasets directly
                print(f"  Copying {dataset_name}: {dataset.shape}")
                out_fold.create_dataset(dataset_name, data=dataset[:])
                
                # Copy attributes
                for attr_name, attr_value in dataset.attrs.items():
                    out_fold[dataset_name].attrs[attr_name] = attr_value
    
    # Copy file-level attributes
    for attr_name, attr_value in in_f.attrs.items():
        out_f.attrs[attr_name] = attr_value

print(f"\nResizing complete! Output saved to: {output_file}")
