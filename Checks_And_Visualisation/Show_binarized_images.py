import numpy as np
import h5py
import matplotlib.pyplot as plt

# Same binarization function as used in the Tsetlin Machine code
def binarize_images(X, n_bits):
    n_samples, x_dim, y_dim = X.shape
    vmin = np.quantile(X, 0.001)
    vmax = np.quantile(X, 0.999)
    X =  ((X - vmin) / (vmax - vmin)).clip(0, 1)
    X = X.reshape(n_samples, -1)
    thresholds = np.linspace(0, 1-(1/n_bits), n_bits)
    x_bin = (X[...,np.newaxis] > thresholds).astype(np.uint32)
    return x_bin.reshape(X.shape[0], x_dim, y_dim, n_bits)

                             
FOLD = 0                                # which fold to visualize
image_nr = 450                          # which image to visualize
n_bits = 8                              # number of bits for binarization

with h5py.File(r'path/to/your/dataset.h5', 'r') as f:
    images = f[f'fold_{FOLD}']['image']
    idx = f[f'fold_{FOLD}']['patient_idx']
    diagnosis = f[f'fold_{FOLD}']['diagnosis']
    target = f[f'fold_{FOLD}']['target']
    
    # Binarize the image
    img = images[image_nr]
    img_bin = binarize_images(img[np.newaxis, ...], n_bits=n_bits)[0]

    # Display all n binary channels in a grid of n_bits/4 rows and 4 columns
    n_rows = n_bits // 4
    plt.figure(figsize=(20, 5 * n_rows))
    for i in range(n_bits):
        plt.subplot(n_rows, 4, i + 1)
        plt.imshow(img_bin[..., i], cmap='gray')
        plt.axis("off")
    plt.suptitle(f"Image {image_nr} in fold {FOLD}, Patient Index {idx[image_nr]}, Diagnosis {diagnosis[image_nr]}, Target {target[image_nr]}, Size {img.shape}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()