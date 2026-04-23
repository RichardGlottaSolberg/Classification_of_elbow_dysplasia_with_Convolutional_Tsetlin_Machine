import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --- Configure your datasets here ---
# Format: "label_name": "path/to/log.txt"
# The key becomes the label in the plot.

datasets = {
    "Perfekt":              r"path\to\perfekt_log.txt",
    "Rotert":               r"path\to\rotated_log.txt",
    "Speilet":              r"path\to\mirrored_log.txt",
    "Begge":                r"path\to\both_log.txt",
}

COLORS = ["#5094D8", '#1D9E75', "#C5C51F", "#AA2671"]

# ------------------------------------

def parse_log(filepath):
    epochs, train_mcc, val_mcc = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('|')]
            if len(parts) < 12:
                continue
            try:
                epochs.append(int(parts[0]))
                train_mcc.append(float(parts[3]))
                val_mcc.append(float(parts[9]))
            except ValueError:
                continue
    return epochs, train_mcc, val_mcc


fig, ax = plt.subplots(figsize=(11, 6))

for i, (label, filepath) in enumerate(datasets.items()):
    epochs, train_mcc, val_mcc = parse_log(filepath)
    color = COLORS[i]
    ax.plot(epochs, train_mcc, color=color, linewidth=2, linestyle='-')
    ax.plot(epochs, val_mcc,   color=color, linewidth=2, linestyle='--')

ax.set_xlabel('Epoker', fontsize=16)
ax.set_ylabel('MCC', fontsize=16)
ax.set_title('MCC — Trening og Validering', fontsize=16)
ax.grid(True, linestyle='--', linewidth=0.8, color='#ddd')

legend_handles = [
    mlines.Line2D([], [], color=COLORS[i], linewidth=2, label=label)
    for i, label in enumerate(datasets)
]
legend_handles += [
    mlines.Line2D([], [], color='#888', linewidth=2, linestyle='-',  label='Train'),
    mlines.Line2D([], [], color='#888', linewidth=2, linestyle='--', label='Val'),
]
ax.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig('mcc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
