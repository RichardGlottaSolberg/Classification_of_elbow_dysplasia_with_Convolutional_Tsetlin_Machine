import os
import numpy as np
import h5py
import json
import pickle
from bz2 import BZ2File
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import argparse

def binarize_images(X, n_bits):
    """Binarize images using quantile-based normalization"""
    n_samples, x_dim, y_dim, _ = X.shape
    vmin = np.quantile(X, 0.001)
    vmax = np.quantile(X, 0.999)
    X = ((X - vmin) / (vmax - vmin)).clip(0, 1)
    X = X.reshape(n_samples, -1)
    thresholds = np.linspace(0, 1-(1/n_bits), n_bits)
    x_bin = (X[..., np.newaxis] > thresholds).astype(np.uint32)
    return x_bin.reshape(X.shape[0], x_dim, y_dim, n_bits)

def load_model_and_config(exp_name):
    """Load trained model and its configuration"""
    base_path = f'{project_dir}/perf/{exp_name}/'
    
    # Load configuration
    with open(f'{base_path}best_configs.json', 'r') as f:
        config = json.load(f)
    
    # Load model
    with BZ2File(f'{base_path}best_model.tm', "rb") as file:
        model_state = pickle.load(file)
    
    # Extract model parameters
    model_params = {k: v for k, v in config.items() if k != 'dataset_path' and k != 'n_bits'}
    tm = MultiClassConvolutionalTsetlinMachine2D(**model_params)
    tm.load(model_state)
    
    return tm, config

def get_test_data(dataset_path):
    """Load test data"""
    with h5py.File(dataset_path, 'r') as f:
        X_test = f['fold_5']['image'][:]
        Y_test = f['fold_5']['target'][:]
        try:
            patient_idx = f['fold_5']['patient_idx'][:]
        except:
            patient_idx = np.arange(len(Y_test))
    return X_test, Y_test, patient_idx

def get_predictions(tm, X_test_bin):
    """Get predictions and clause outputs"""
    predictions = tm.predict(X_test_bin)
    return predictions

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300)
    plt.close()
    
    return cm

def plot_roc_curve(y_true, y_pred, save_path):
    """Plot ROC curve for binary classification"""
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curve.png', dpi=300)
        plt.close()

def plot_precision_recall_curve(y_true, y_pred, save_path):
    """Plot Precision-Recall curve"""
    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/precision_recall_curve.png', dpi=300)
        plt.close()

def visualize_misclassified_samples(X_test, y_true, y_pred, patient_idx, 
                                    save_path, n_samples=20):
    """Visualize misclassified samples"""
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\nTotal misclassified samples: {len(misclassified_idx)}")
    
    # Select random subset if too many
    if len(misclassified_idx) > n_samples:
        selected_idx = np.random.choice(misclassified_idx, n_samples, replace=False)
    else:
        selected_idx = misclassified_idx
    
    # Create grid of misclassified images
    n_cols = 5
    n_rows = int(np.ceil(len(selected_idx) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(selected_idx):
            sample_idx = selected_idx[idx]
            img = X_test[sample_idx].squeeze()
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'ID: {patient_idx[sample_idx]}\n'
                        f'True: {y_true[sample_idx]}, Pred: {y_pred[sample_idx]}',
                        fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/misclassified_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed information about misclassifications
    with open(f'{save_path}/misclassified_details.txt', 'w') as f:
        f.write("Misclassified Samples Details\n")
        f.write("=" * 50 + "\n\n")
        for idx in misclassified_idx:
            f.write(f"Patient ID: {patient_idx[idx]}\n")
            f.write(f"True Label: {y_true[idx]}\n")
            f.write(f"Predicted Label: {y_pred[idx]}\n")
            f.write("-" * 30 + "\n")

def analyze_per_class_performance(y_true, y_pred, save_path):
    """Detailed per-class performance analysis"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    with open(f'{save_path}/classification_report.txt', 'w') as f:
        f.write("Detailed Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(classification_report(y_true, y_pred))
    
    # Visualize per-class metrics
    classes = [k for k in report.keys() if k.isdigit()]
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report[c][metric] for c in classes]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/per_class_metrics.png', dpi=300)
    plt.close()

def extract_clause_weights(tm, save_path):
    """Extract and visualize clause weights/importance"""
    try:
        # Get clause outputs for analysis
        # Note: Actual implementation depends on PySparseCoalescedTsetlinMachineCUDA API
        print("\nExtracting clause information...")
        
        # This is a placeholder - adjust based on actual TM API
        clause_info = {
            'number_of_clauses': tm.number_of_clauses,
            'number_of_classes': tm.number_of_classes if hasattr(tm, 'number_of_classes') else 2
        }
        
        with open(f'{save_path}/clause_info.json', 'w') as f:
            json.dump(clause_info, f, indent=4)
        
        print(f"Clause information saved to {save_path}/clause_info.json")
        
    except Exception as e:
        print(f"Could not extract clause weights: {e}")

def compare_correct_vs_incorrect(X_test, y_true, y_pred, save_path):
    """Compare statistics between correctly and incorrectly classified samples"""
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask
    
    X_correct = X_test[correct_mask]
    X_incorrect = X_test[incorrect_mask]
    
    # Compute statistics
    stats = {
        'correct': {
            'mean': float(np.mean(X_correct)),
            'std': float(np.std(X_correct)),
            'min': float(np.min(X_correct)),
            'max': float(np.max(X_correct)),
            'count': int(np.sum(correct_mask))
        },
        'incorrect': {
            'mean': float(np.mean(X_incorrect)),
            'std': float(np.std(X_incorrect)),
            'min': float(np.min(X_incorrect)),
            'max': float(np.max(X_incorrect)),
            'count': int(np.sum(incorrect_mask))
        }
    }
    
    with open(f'{save_path}/correct_vs_incorrect_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Visualize intensity distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(X_correct.flatten(), bins=50, alpha=0.7, label='Correct', color='green')
    axes[0].hist(X_incorrect.flatten(), bins=50, alpha=0.7, label='Incorrect', color='red')
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pixel Intensity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Average images
    avg_correct = np.mean(X_correct, axis=0).squeeze()
    avg_incorrect = np.mean(X_incorrect, axis=0).squeeze()
    
    im1 = axes[1].imshow(np.hstack([avg_correct, avg_incorrect]), cmap='gray')
    axes[1].set_title('Average Images: Correct (Left) vs Incorrect (Right)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/correct_vs_incorrect_analysis.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Tsetlin Machine Results')
    parser.add_argument('--exp_name', required=True, type=str, 
                       help='Experiment name to analyze')
    parser.add_argument('--n_misclassified', default=20, type=int,
                       help='Number of misclassified samples to visualize')
    args = parser.parse_args()
    
    exp_name = args.exp_name
    project_dir = 'path_to_your_project_directory'                      # Update this to your actual project directory
    analysis_path = f'{project_dir}/perf/{exp_name}/analysis'
    os.makedirs(analysis_path, exist_ok=True)
    
    print(f"Analyzing experiment: {exp_name}")
    print("=" * 60)
    
    # Load model and config
    print("\nLoading model and configuration...")
    tm, config = load_model_and_config(exp_name)
    dataset_path = config['dataset_path']
    n_bits = config['n_bits']
    
    # Load test data
    print("Loading test data...")
    X_test, Y_test, patient_idx = get_test_data(dataset_path)
    
    if len(X_test.shape) == 3:
        X_test = X_test[..., np.newaxis]
    
    # Binarize test data
    print("Binarizing images...")
    X_test_bin = binarize_images(X_test, n_bits).reshape(X_test.shape[0], -1).astype(np.uint32)
    
    # Get predictions
    print("Getting predictions...")
    y_pred = get_predictions(tm, X_test_bin)
    
    # Generate comprehensive analysis
    print("\nGenerating analysis...")
    
    # 1. Confusion Matrix
    print("- Plotting confusion matrix...")
    cm = plot_confusion_matrix(Y_test, y_pred, analysis_path)
    
    # 2. ROC and PR curves (for binary classification)
    print("- Plotting ROC curve...")
    plot_roc_curve(Y_test, y_pred, analysis_path)
    
    print("- Plotting Precision-Recall curve...")
    plot_precision_recall_curve(Y_test, y_pred, analysis_path)
    
    # 3. Per-class performance
    print("- Analyzing per-class performance...")
    analyze_per_class_performance(Y_test, y_pred, analysis_path)
    
    # 4. Visualize misclassified samples
    print("- Visualizing misclassified samples...")
    visualize_misclassified_samples(X_test, Y_test, y_pred, patient_idx,
                                   analysis_path, n_samples=args.n_misclassified)
    
    # 5. Compare correct vs incorrect predictions
    print("- Comparing correct vs incorrect predictions...")
    compare_correct_vs_incorrect(X_test, Y_test, y_pred, analysis_path)
    
    # 6. Extract clause information (explainability)
    print("- Extracting clause information...")
    extract_clause_weights(tm, analysis_path)
    
    # Summary statistics
    accuracy = np.mean(Y_test == y_pred)
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total Samples: {len(Y_test)}")
    print(f"Correctly Classified: {np.sum(Y_test == y_pred)}")
    print(f"Misclassified: {np.sum(Y_test != y_pred)}")
    print(f"\nAll results saved to: {analysis_path}")
