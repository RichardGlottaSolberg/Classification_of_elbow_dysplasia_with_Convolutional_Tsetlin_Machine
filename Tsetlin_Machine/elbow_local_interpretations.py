import argparse
import json
import pickle
from bz2 import BZ2File
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from tqdm import tqdm

# Colour palette
icefire = sns.color_palette("icefire", as_cmap=True)
# Create custom diverging colormap: dark red -> white -> dark blue
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "red_white_blue",
    ["#8B0000", "#FFFFFF", "#00008B"]
)

CLASS_NAMES = ["Normal", "Abnormal"]   # label 0 / 1


# Model loading

def load_model(model_path: str, configs_path: str) -> MultiClassConvolutionalTsetlinMachine2D:
    """Reconstruct the TM from best_configs.json and load its saved state."""
    with open(configs_path, "r") as f:
        config = json.load(f)

    params = {
        "number_of_clauses": config["number_of_clauses"],
        "T":                 config["T"],
        "s":                 config["s"],
        "dim":               tuple(config["dim"]),
        "patch_dim":         tuple(config["patch_dim"]),
    }

    tm = MultiClassConvolutionalTsetlinMachine2D(**params)

    with BZ2File(model_path, "rb") as f:
        state_dict = pickle.load(f)

    tm.load(state_dict)
    return tm


# Binarisation

def binarize_images(X: np.ndarray, n_bits: int) -> np.ndarray:
    """ Same binarization function as in main pipeline"""
    n_samples, x_dim, y_dim, _ = X.shape
    vmin = np.quantile(X, 0.001)
    vmax = np.quantile(X, 0.999)
    X = ((X - vmin) / (vmax - vmin)).clip(0, 1)
    X = X.reshape(n_samples, -1)
    thresholds = np.linspace(0, 1 - (1 / n_bits), n_bits)
    x_bin = (X[..., np.newaxis] > thresholds).astype(np.uint32)
    return x_bin.reshape(n_samples, x_dim, y_dim, n_bits)


def unbinarize(x_bin: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Reconstruct a displayable grayscale image from a binarized array.
    x_bin : (H, W, n_bits)
    Returns (H, W)  float in [0, 1]
    """
    return x_bin.sum(axis=-1) / n_bits


# Data loading

def load_fold_data(dataset_path: str, fold: int):
    """Load raw (unbinarized) images and labels from a specific fold."""
    with h5py.File(dataset_path, "r") as f:
        X = f[f"fold_{fold}"]["image"][:]
        Y = f[f"fold_{fold}"]["target"][:]
    return X, Y


# Core local transform

def transform_Xs(tm: MultiClassConvolutionalTsetlinMachine2D, Xs_bin_flat: np.ndarray, img_shape: tuple,) -> np.ndarray:
    """
    Compute per-image clause activation maps.

    Parameters
    ----------
    tm           : trained TM
    Xs_bin_flat  : (N, H*W*n_bits) uint32 binarized & flattened images, exactly as fed to tm.predict() and tm.fit()
    img_shape    : (H, W)  spatial shape of the original image

    Returns
    -------
    transformed  : (N, 2, H, W)
                   axis-1 index 0 = positive evidence map
                   axis-1 index 1 = negative evidence map
    """
    num_samples  = Xs_bin_flat.shape[0]
    num_patch_x  = tm.dim[0] - tm.patch_dim[0] + 1
    num_patch_y  = tm.dim[1] - tm.patch_dim[1] + 1
    half_lits    = tm.number_of_features // 2

    # Patchwise activation: which patches fired for each clause on each sample
    # Shape: (N, n_clauses, num_patch_x, num_patch_y)
    co_patchwise = (
        tm.transform_patchwise(Xs_bin_flat)
        .toarray()
        .reshape(num_samples, tm.number_of_clauses, num_patch_x, num_patch_y)
    )

    literals = tm.get_literals()
    weights  = tm.get_weights()   # shape: (n_outputs, n_clauses)

    # Debug: Print actual feature organization
    print(f"Debug info:")
    print(f"  literals shape: {literals.shape}")
    print(f"  tm.number_of_features: {tm.number_of_features}")
    print(f"  half_lits: {half_lits}")
    print(f"  patch_dim: {tm.patch_dim}")
    print(f"  num_patch_x * num_patch_y: {num_patch_x * num_patch_y}")

    # The pixel-level literals are the last tm.patch_dim[0] * tm.patch_dim[1] features
    # Positive literals: from 0 to half_lits
    # Negative literals: from half_lits to end
    # Within each half, the last tm.patch_dim[0] * tm.patch_dim[1] are pixel-level
    
    patch_size = tm.patch_dim[0] * tm.patch_dim[1]
    
    # Extract pixel-level positive/negative literals per clause
    # Shape: (n_clauses, patch_h, patch_w) - binary mask of active features
    positive_literals = (
        literals[:, half_lits - patch_size:half_lits]
        .reshape(tm.number_of_clauses, *tm.patch_dim)
        .astype(np.float32)
    )
    negative_literals = (
        literals[:, tm.number_of_features - patch_size:tm.number_of_features]
        .reshape(tm.number_of_clauses, *tm.patch_dim)
        .astype(np.float32)
    )

    transformed = np.zeros((num_samples, 2, *img_shape))

    for e in tqdm(range(num_samples), desc="Local transform"):
        # Use the predicted class to index into the weight matrix
        pred_class = int(tm.predict(Xs_bin_flat[e:e+1])[0])

        for ci in range(tm.number_of_clauses):
            if weights[pred_class, ci] <= 0:
                continue

            # Find patches where this clause fired on this image
            active_patches = np.argwhere(co_patchwise[e, ci] > 0)
            if len(active_patches) == 0:
                continue

            timg = np.zeros((2, *img_shape))
            for m, n in active_patches:
                timg[0, m:m + tm.patch_dim[0], n:n + tm.patch_dim[1]] += positive_literals[ci]
                timg[1, m:m + tm.patch_dim[0], n:n + tm.patch_dim[1]] += negative_literals[ci]

            transformed[e] += timg * weights[pred_class, ci]

    return transformed


# Plotting

def plot_local(X_raw: np.ndarray, X_bin: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray, transformed: np.ndarray, n_bits: int,) -> plt.Figure:
    """
    Two-row figure:
      Row (a) is original X-ray (reconstructed from binarized for consistency)
      Row (b) is clause activation map overlaid as custom red-white-blue heatmap

    Titles show true label and predicted label; a red border flags mispredictions.

    Parameters
    ----------
    X_raw       : (N, H, W, 1)  original float images  (used for display)
    X_bin       : (N, H, W, n_bits)  binarized images   (used for unbinarize)
    Y_true      : (N,)  ground-truth labels
    Y_pred      : (N,)  model predictions
    transformed : (N, 2, H, W)  output of transform_Xs
    n_bits      : binarization depth, used to scale unbinarize output
    """
    num_samples = X_raw.shape[0]

    fig, axs = plt.subplots(
        2, num_samples,
        figsize=(2.8 * num_samples, 6),
        squeeze=False,
        layout="compressed",
    )

    for ax in axs.ravel():
        ax.axis("off")

    for i in range(num_samples):
        ax_img = axs[0, i]
        ax_map = axs[1, i]

        # Row (a): original X-ray
        # Display the raw image (squeeze channel dim for grayscale)
        display_img = X_raw[i, ..., 0] if X_raw.shape[-1] == 1 else X_raw[i]
        # Normalise to [0,1] for display
        dmin, dmax = display_img.min(), display_img.max()
        display_img = (display_img - dmin) / (dmax - dmin + 1e-7)
        ax_img.imshow(display_img, cmap="gray", vmin=0, vmax=1)

        # Title: true / predicted labels; red border on misprediction
        # Convert predictions to int for indexing
        true_label = int(Y_true[i])
        pred_label = int(Y_pred[i])
        correct = true_label == pred_label
        title_color = "black" if correct else "red"
        ax_img.set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}",
            fontsize=8,
            color=title_color,
        )
        if not correct:
            for spine in ax_img.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
                spine.set_visible(True)

        # Row (b): clause activation map
        diff = transformed[i, 0] - transformed[i, 1]

        # Find symmetric limits for better diverging colormap
        vmax = max(abs(diff.min()), abs(diff.max()))
        
        # Use TwoSlopeNorm to center white at 0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax_map.imshow(diff, cmap=custom_cmap, norm=norm)

    # Row labels
    for row_idx, label in enumerate(["(a) Input X-ray", "(b) TM clause map"]):
        axs[row_idx, 0].annotate(
            label,
            xy=(0, 0.5),
            xytext=(-8, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
            fontsize=9,
        )

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axs[1, :], fraction=0.015, pad=0.01, aspect=30)
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels(["Negative", "Neutral", "Positive"])
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_label_position("right")
    # Adjust label rotation and alignment
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_ha("left")

    fig.suptitle(
        "Local Tsetlin Machine Interpretation — Canine Elbow Dysplasia",
        fontsize=11,
        y=1.01,
    )

    return fig


# Use

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local TM interpretations for canine elbow dysplasia"
    )
    parser.add_argument("--exp_name",     type=str, default=None,
                        help="Experiment name (folder inside --base_dir)")
    parser.add_argument("--model_path",   type=str, default=None,
                        help="Direct path to best_model.tm (overrides --exp_name)")
    parser.add_argument("--configs_path", type=str, default=None,
                        help="Direct path to best_configs.json (overrides --exp_name)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Direct path to the HDF5 dataset (overrides value in configs)")
    parser.add_argument("--out_path",     type=str, default=None,
                        help="Output figure path (PDF or PNG). Default: <exp_dir>/local_inter.pdf")
    parser.add_argument("--base_dir",     type=str, default="D:/CubiAI_tsetlin/perf",
                        help="Base directory for experiments")
    parser.add_argument("--fold",         type=int, default=4,
                        help="Which fold to draw samples from (default: 4, the val fold)")
    parser.add_argument("--n_per_class",  type=int, default=3,
                        help="Number of samples to show per class (default: 3)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    # Resolve paths
    if args.model_path and args.configs_path:
        model_path   = args.model_path
        configs_path = args.configs_path
        out_path     = args.out_path or str(Path(args.model_path).parent / "local_inter.pdf")
    elif args.exp_name:
        exp_dir      = Path(args.base_dir) / args.exp_name
        model_path   = args.model_path   or str(exp_dir / "best_model.tm")
        configs_path = args.configs_path or str(exp_dir / "best_configs.json")
        out_path     = args.out_path     or str(exp_dir / "local_inter.pdf")
    else:
        parser.error("Provide either --exp_name or both --model_path and --configs_path")

    # Load model & config
    print(f"Loading model from  : {model_path}")
    print(f"Loading configs from: {configs_path}")
    tm = load_model(model_path, configs_path)

    with open(configs_path) as f:
        cfg = json.load(f)

    n_bits       = cfg["n_bits"]
    dataset_path = args.dataset_path or cfg["dataset_path"]
    img_h, img_w = tm.dim[0], tm.dim[1]

    print(f"\nModel summary:")
    print(f"  Image dim  : {tm.dim}  (H x W x n_bits)")
    print(f"  Patch dim  : {tm.patch_dim}")
    print(f"  Clauses    : {tm.number_of_clauses}")
    print(f"  n_bits     : {n_bits}")

    # Load data
    print(f"\nLoading fold {args.fold} from: {dataset_path}")
    X_raw, Y = load_fold_data(dataset_path, fold=args.fold)

    # Ensure channel dimension exists: (N, H, W) → (N, H, W, 1)
    if len(X_raw.shape) == 3:
        X_raw = X_raw[..., np.newaxis]

    # Sample selection: n_per_class from each class
    rng = np.random.default_rng(args.seed)
    selected_idx = []
    for cls in range(2):   # 0 = Normal, 1 = Abnormal
        inds = np.argwhere(Y == cls).ravel()
        if len(inds) < args.n_per_class:
            print(f"  Warning: only {len(inds)} samples for class {CLASS_NAMES[cls]}, using all.")
            chosen = inds
        else:
            chosen = rng.choice(inds, args.n_per_class, replace=False)
        selected_idx.extend(chosen.tolist())

    selected_idx = np.array(selected_idx)
    print(f"\nSelected {len(selected_idx)} samples "
          f"({args.n_per_class} x Normal, {args.n_per_class} x Abnormal)")

    X_sel     = X_raw[selected_idx]      # (N, H, W, 1)  raw for display
    Y_sel     = Y[selected_idx]          # (N,)

    # Binarize
    print("Binarizing selected images...")
    X_bin = binarize_images(X_sel, n_bits)                               # (N, H, W, n_bits)
    X_bin_flat = X_bin.reshape(len(selected_idx), -1).astype(np.uint32)  # (N, H*W*n_bits)

    # Predict
    Y_pred = tm.predict(X_bin_flat)

    correct = np.sum(Y_pred == Y_sel)
    print(f"Accuracy on selected samples: {correct}/{len(selected_idx)}")

    # Local transform
    print("\nComputing local clause maps...")
    transformed = transform_Xs(tm, X_bin_flat, img_shape=(img_h, img_w))

    # Plot
    print("Plotting...")
    fig = plot_local(X_sel, X_bin, Y_sel, Y_pred, transformed, n_bits)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.show()
