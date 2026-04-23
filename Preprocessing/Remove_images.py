import h5py
import numpy as np

file = r'path\to\your\input_dataset.h5'             # Change this to the path of dataset you want to clean
output_file = r'path\to\your\output_dataset.h5'     # Change this to desired path and name of the cleaned dataset
remove_patient_ids = [6712, 3738]                   # patient ids to remove from the dataset (these are the ones with bad images in fold_2 and fold_3)

folds_to_clean = ['fold_2', 'fold_3']
folds_to_copy = ['fold_0', 'fold_1', 'fold_4', 'fold_5']

with h5py.File(file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
    # Copy the folds that we want to keep without modification
    for fold in folds_to_copy:
        print(f"Copying {fold}: Copying directly (no changes)")
        f_in.copy(fold, f_out)
    
    # Process the folds that we want to clean
    for fold in folds_to_clean:
        patient_ids = f_in[fold]['patient_idx'][:]
        
        keep_mask = np.isin(patient_ids, remove_patient_ids, invert=True)
        keep_indices = np.where(keep_mask)[0]
        
        fold_group = f_out.create_group(fold)
        
        fold_group.create_dataset('image', data=f_in[fold]['image'][keep_indices])
        fold_group.create_dataset('target', data=f_in[fold]['target'][keep_indices])
        fold_group.create_dataset('diagnosis', data=f_in[fold]['diagnosis'][keep_indices])
        fold_group.create_dataset('patient_idx', data=f_in[fold]['patient_idx'][keep_indices])
        
        remove_count = len(patient_ids) - len(keep_indices)
        print(f"\n{fold}: kept {len(keep_indices)}, removed {remove_count}")
        
print(f"\nOriginal kept as: {file}")
print(f"Cleaned saved as: {output_file}")