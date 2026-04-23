import argparse
import json
import pickle
from bz2 import BZ2File
from pathlib import Path

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

CLASS_NAMES = ["Normal", "Abnormal"]   # matches label in the dataset: 0 for normal, 1 for abnormal


# Model loading

def load_model(model_path: str, configs_path: str) -> MultiClassConvolutionalTsetlinMachine2D:
    """
    Reconstruct the TM from best_configs.json and load its saved state.

    best_configs.json contains:
        number_of_clauses, T, s, dim (list), patch_dim (list),
        dataset_path, n_bits
    """
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

    # best_model.tm stores tm.save() directly (not wrapped in a checkpoint dict)
    tm.load(state_dict)
    return tm


# Position helpers

def clause_positions_pos(lits: np.ndarray) -> np.ndarray:
    """
    For each clause, find the *last* active literal in the x- and y-position
    sub-vectors (inclusive upper bound for the positive position window).
    """
    pos_len = lits.shape[1] // 2
    xpos = lits[:, :pos_len]
    ypos = lits[:, pos_len:]

    ret = np.zeros((lits.shape[0], 2), dtype=int)
    for ci in range(lits.shape[0]):
        nnz = np.argwhere(xpos[ci]).ravel()
        ret[ci, 0] = nnz[-1] + 1 if len(nnz) > 0 else 0
        nnz = np.argwhere(ypos[ci]).ravel()
        ret[ci, 1] = nnz[-1] + 1 if len(nnz) > 0 else 0
    return ret


def clause_positions_neg(lits: np.ndarray) -> np.ndarray:
    """
    For each clause, find the *first* active literal in the x- and y-position
    sub-vectors (inclusive upper bound for the negative position window).
    """
    pos_len = lits.shape[1] // 2
    xpos = lits[:, :pos_len]
    ypos = lits[:, pos_len:]

    ret = np.zeros((lits.shape[0], 2), dtype=int)
    for ci in range(lits.shape[0]):
        nnz = np.argwhere(xpos[ci]).ravel()
        ret[ci, 0] = nnz[0] if len(nnz) > 0 else 0
        nnz = np.argwhere(ypos[ci]).ravel()
        ret[ci, 1] = nnz[0] if len(nnz) > 0 else 0
    return ret


# Literal value helpers (multi-bit grayscale)

def pixel_values_pos(a: np.ndarray, levels: int) -> np.ndarray:
    """
    Decode positive literals back to a scalar pixel value in [0, 1].
    a: shape (n_clauses, patch_h, patch_w, levels) one channel, grayscale
    Returns shape (n_clauses, patch_h, patch_w, 1)
    """
    color_channels = a.shape[-1] // levels

    def decode(b):
        ret = np.empty(color_channels)
        for i in range(color_channels):
            nz = np.argwhere(b[i * levels:(i + 1) * levels]).ravel()
            ret[i] = (nz[-1] + 1) if len(nz) > 0 else 0
        return ret

    return np.apply_along_axis(decode, -1, a) / (levels + 1)


def pixel_values_neg(a: np.ndarray, levels: int) -> np.ndarray:
    """
    Decode negative literals back to a scalar pixel value in [0, 1].
    """
    color_channels = a.shape[-1] // levels

    def decode(b):
        ret = np.empty(color_channels)
        for i in range(color_channels):
            nz = np.argwhere(b[i * levels:(i + 1) * levels]).ravel()
            ret[i] = nz[0] if len(nz) > 0 else 0
        return ret

    return np.apply_along_axis(decode, -1, a) / (levels + 1)


# Core transform

def global_transform(tm: MultiClassConvolutionalTsetlinMachine2D) -> np.ndarray:
    """
    Build a (num_classes, 2, img_h, img_w, 1) accumulation array where:
        axis-1 index 0 = positive evidence contributions
        axis-1 index 1 = negative evidence contributions

    Adapted from global_interpretations.py for:
        - 128x128 grayscale images (1 colour channel)
        - n_bits binarization levels stored in tm.dim[2]
        - MultiClassConvolutionalTsetlinMachine2D (binary / multi-class)
    """
    img_h, img_w = tm.dim[0], tm.dim[1]
    levels       = tm.dim[2]          # = n_bits used during training
    color_channels = 1                # grayscale

    num_patch_x = img_h - tm.patch_dim[0] + 1
    num_patch_y = img_w - tm.patch_dim[1] + 1
    half_lits   = tm.number_of_features // 2

    literals      = tm.get_literals()
    weights       = tm.get_weights()
    patch_weights = tm.get_patch_weights()

    # Normalise patch weights per clause; zero out low-confidence patches
    patch_weights = patch_weights / (patch_weights.max(axis=(1, 2), keepdims=True) + 1e-7)
    threshold = 0.2
    patch_weights[patch_weights < threshold] = 0.0

    # Extract pixel-level literals
    pos_lit_raw = (
        literals[:, (num_patch_x - 1) + (num_patch_y - 1):half_lits]
        .reshape(tm.number_of_clauses, *tm.patch_dim, tm.dim[2])
        .astype(np.float32)
    )
    neg_lit_raw = (
        literals[:, half_lits + (num_patch_x - 1) + (num_patch_y - 1):]
        .reshape(tm.number_of_clauses, *tm.patch_dim, tm.dim[2])
        .astype(np.float32)
    )

    positive_literals = pixel_values_pos(pos_lit_raw, levels)   # → (n_clauses, ph, pw, 1)
    negative_literals = pixel_values_neg(neg_lit_raw, levels)   # → (n_clauses, ph, pw, 1)

    # Extract clause spatial positions 
    positive_positions = clause_positions_pos(
        literals[:, :(num_patch_x - 1) + (num_patch_y - 1)]
    )
    negative_positions = clause_positions_neg(
        literals[:, half_lits:half_lits + (num_patch_x - 1) + (num_patch_y - 1)]
    )

    # Accumulate clause contributions into the global interpretation map
    transformed = np.zeros((tm.number_of_outputs, 2, img_h, img_w, color_channels))

    for c in range(tm.number_of_outputs):
        for ci in tqdm(range(tm.number_of_clauses), leave=False, desc=f"Class {CLASS_NAMES[c]}"):
            if weights[c, ci] <= 0:
                continue

            timg = np.zeros((2, img_h, img_w, color_channels))

            # Positive positions: iterate from start position to end of valid range
            sx, sy = positive_positions[ci]
            for i in range(sx, num_patch_x):
                for j in range(sy, num_patch_y):
                    if patch_weights[ci, i, j] > 0:
                        timg[0, i:i + tm.patch_dim[0], j:j + tm.patch_dim[1]] += (
                            patch_weights[ci, i, j] * positive_literals[ci]
                        )

            # Negative positions: iterate from 0 up to start position
            sx, sy = negative_positions[ci]
            for i in range(0, sx + 1):
                for j in range(0, sy + 1):
                    if patch_weights[ci, i, j] > 0:
                        timg[1, i:i + tm.patch_dim[0], j:j + tm.patch_dim[1]] += (
                            patch_weights[ci, i, j] * negative_literals[ci]
                        )

            transformed[c] += timg * weights[c, ci]

    return transformed


# Plotting

def plot_transformed(transformed: np.ndarray, class_names: list[str]) -> plt.Figure:
    """
    Produce a side-by-side plot of the global interpretation for each class.

    For grayscale output the difference map (positive - negative contributions)
    is rendered with a custom red-white-blue diverging colourmap: dark red = negative evidence,
    white = neutral, dark blue = positive evidence.
    """
    num_classes = transformed.shape[0]
    color_channels = transformed.shape[-1]   # 1 for grayscale

    fig, axes = plt.subplots(
        1, num_classes + 1,
        figsize=(4 * num_classes + 1, 4.5),
        gridspec_kw={"width_ratios": [1] * num_classes + [0.06]},
    )

    im_ref = None
    for c in range(num_classes):
        ax = axes[c]
        ax.axis("off")

        # difference: positive evidence minus negative evidence
        img = transformed[c, 0] - transformed[c, 1]

        if color_channels == 1:
            # Squeeze to 2-D and plot as a diverging heatmap
            img_2d = img[..., 0]

            # Find symmetric limits for better diverging colormap
            vmax = max(abs(img_2d.min()), abs(img_2d.max()))

            # Use TwoSlopeNorm to center white at 0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(img_2d, cmap=custom_cmap, norm=norm)
            if im_ref is None:
                im_ref = im
        else:
            # RGB path (kept for completeness, not used for this dataset)
            for ch in range(color_channels):
                t = img[..., ch]
                vmax = max(abs(t.min()), abs(t.max()))
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                im = ax.imshow(t, cmap=custom_cmap, norm=norm)
                if im_ref is None:
                    im_ref = im

        ax.set_title(class_names[c], fontsize=13, fontweight="bold")

    # Colourbar
    cax = axes[-1]
    cax.axis("on")
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, cax=cax, fraction=1, pad=0)
        cbar.set_ticks([-vmax, 0, vmax])
        cbar.set_ticklabels(["Negative", "Neutral", "Positive"])
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.yaxis.set_label_position("right")
        # Adjust label spacing and alignment
        for label in cbar.ax.get_yticklabels():
            label.set_rotation(0)
            label.set_ha("left")
    else:
        cax.axis("off")

    fig.suptitle("Global Tsetlin Machine Interpretation\nCanine Elbow Dysplasia", fontsize=12)
    fig.tight_layout()
    return fig


# Use

def main():
    parser = argparse.ArgumentParser(description="Global TM interpretations for canine elbow dysplasia")
    parser.add_argument("--exp_name",    type=str, default=None,
                        help="Experiment name used in tsetlin_optuna_on_one_fold.py")
    parser.add_argument("--model_path",  type=str, default=None,
                        help="Direct path to best_model.tm  (overrides --exp_name default)")
    parser.add_argument("--configs_path",type=str, default=None,
                        help="Direct path to best_configs.json (overrides --exp_name default)")
    parser.add_argument("--out_path",    type=str, default=None,
                        help="Where to save the figure (PDF/PNG).  Default: <exp_dir>/global_inter.pdf")
    parser.add_argument("--base_dir",    type=str, default="D:/CubiAI_tsetlin/perf",
                        help="Base directory for experiments (default: D:/CubiAI_tsetlin/perf)")
    args = parser.parse_args()

    # Resolve paths 
    if args.model_path and args.configs_path:
        model_path   = args.model_path
        configs_path = args.configs_path
        out_path     = args.out_path or str(Path(args.model_path).parent / "global_inter.pdf")
    elif args.exp_name:
        exp_dir      = Path(args.base_dir) / args.exp_name
        model_path   = args.model_path   or str(exp_dir / "best_model.tm")
        configs_path = args.configs_path or str(exp_dir / "best_configs.json")
        out_path     = args.out_path     or str(exp_dir / "global_inter.pdf")
    else:
        parser.error("Provide either --exp_name or both --model_path and --configs_path")

    print(f"Loading model from : {model_path}")
    print(f"Loading configs from: {configs_path}")

    # Load & run 
    tm = load_model(model_path, configs_path)

    print(f"\nModel summary:")
    print(f"  Clauses    : {tm.number_of_clauses}")
    print(f"  Image dim  : {tm.dim}  (H x W x n_bits)")
    print(f"  Patch dim  : {tm.patch_dim}")
    print(f"  Outputs    : {tm.number_of_outputs}")

    print("\nComputing global transform (this may take a few minutes for large clause counts)...")
    transformed = global_transform(tm)

    print("Plotting...")
    fig = plot_transformed(transformed, CLASS_NAMES)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.show()
