import numpy as np
import h5py
import matplotlib.pyplot as plt

BATCH = 100                             # change to see more or fewer images at once
FOLD = 0                                # which fold to visualize

# Loop through a fold and show all images in batches, displaying the image and its size
with h5py.File(r'path\to\your\dataset.h5', 'r') as f:
    images = f[f'fold_{FOLD}']['image']
    N = images.shape[0]

    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        batch = images[start:end]       # only BATCH images loaded

        cols = int(np.sqrt(BATCH))
        rows = int(np.ceil(len(batch) / cols))

        plt.figure(figsize=(10, 10))
        for i, img in enumerate(batch):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis("off")

        plt.suptitle(f"Images {start}–{end-1} in fold {FOLD}")
        plt.tight_layout()
        plt.show()
