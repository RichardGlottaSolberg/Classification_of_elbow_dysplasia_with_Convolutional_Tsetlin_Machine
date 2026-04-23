import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
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

def analyse_misclassified_details(txt_file):
    """Analyses misclassified_details.txt and extract information"""
    misclassified_data = []
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    current_sample = {}
    for line in lines:
        line = line.strip()
        if line.startswith('Patient ID:'):
            if current_sample:
                misclassified_data.append(current_sample)
            current_sample = {'patient_id': int(float(line.split(':')[1].strip()))}
        elif line.startswith('True Label:'):
            current_sample['true_label'] = int(float(line.split(':')[1].strip()))
        elif line.startswith('Predicted Label:'):
            current_sample['predicted_label'] = int(float(line.split(':')[1].strip()))
    
    if current_sample:
        misclassified_data.append(current_sample)
    
    return misclassified_data

def get_test_data(dataset_path):
    """Load test data"""
    with h5py.File(dataset_path, 'r') as f:
        X_test = f['fold_5']['image'][:]
        Y_test = f['fold_5']['target'][:]
        try:
            patient_idx = f['fold_5']['patient_idx'][:]
        except:
            patient_idx = np.arange(len(Y_test))
        try:
            diagnosis = f['fold_5']['diagnosis'][:]
        except:
            diagnosis = None
    return X_test, Y_test, patient_idx, diagnosis

def plot_misclassified_distribution(misclassified_data, X_test, Y_test, patient_idx, 
                                   diagnosis, save_path, diagnosis_names=None):
    """Plot distribution of correctly and misclassified samples per diagnosis"""
    
    # Default diagnosis names if not provided
    if diagnosis_names is None:
        diagnosis_names = {0: 'Diagnosis 0', 1: 'Diagnosis 1', 
                          2: 'Diagnosis 2', 3: 'Diagnosis 3'}
    
    # Create set of misclassified patient IDs for quick lookup
    misclassified_patient_ids = set(sample['patient_id'] for sample in misclassified_data)
    
    # Count correct and misclassified per diagnosis
    diagnosis_stats = defaultdict(lambda: {'correct': 0, 'misclassified': 0})
    
    for idx, patient_id in enumerate(patient_idx):
        if diagnosis is not None:
            diag = int(diagnosis[idx])
        else:
            diag = int(Y_test[idx])
        
        if patient_id in misclassified_patient_ids:
            diagnosis_stats[diag]['misclassified'] += 1
        else:
            diagnosis_stats[diag]['correct'] += 1
    
    # Create grouped bar chart
    diagnoses = sorted(diagnosis_stats.keys())
    correct_counts = [diagnosis_stats[d]['correct'] for d in diagnoses]
    misclassified_counts = [diagnosis_stats[d]['misclassified'] for d in diagnoses]
    
    diagnosis_labels = [diagnosis_names[d] for d in diagnoses]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(diagnosis_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, correct_counts, width, label='Correctly Classified',
                   color='#4ECDC4', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, misclassified_counts, width, label='Misclassified',
                   color='#FF6B6B', edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Diagnosis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Classification Performance per Diagnosis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(diagnosis_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/classification_distribution_per_diagnosis.png', dpi=300)
    plt.close()
    
    # Print statistics
    print(f"\nClassification Performance by Diagnosis:")
    print("=" * 60)
    for d in diagnoses:
        correct = diagnosis_stats[d]['correct']
        misclass = diagnosis_stats[d]['misclassified']
        total = correct + misclass
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{diagnosis_names[d]}: Correct={correct}, Misclassified={misclass}, "
              f"Total={total}, Accuracy={accuracy:.1f}%")

def visualize_binarized_misclassified(misclassified_data, X_test, Y_test, patient_idx, 
                                      diagnosis, n_bits, save_path, n_per_class=3):
    """Visualize binarized images of misclassified samples grouped by diagnosis"""
    
    # Create set of misclassified patient IDs
    misclassified_patient_ids = {sample['patient_id']: sample for sample in misclassified_data}
    
    # Group misclassified samples by diagnosis (not target)
    samples_by_diagnosis = defaultdict(list)
    for idx, patient_id in enumerate(patient_idx):
        if patient_id in misclassified_patient_ids:
            sample = misclassified_patient_ids[patient_id]
            diag = int(diagnosis[idx]) if diagnosis is not None else int(Y_test[idx])
            
            samples_by_diagnosis[diag].append({
                'test_idx': idx,
                'patient_id': patient_id,
                'true_label': sample['true_label'],
                'predicted_label': sample['predicted_label'],
                'diagnosis': diag
            })
    
    # Create visualization for each diagnosis
    diagnoses = sorted(samples_by_diagnosis.keys())
    
    for diagnosis in diagnoses:
        samples = samples_by_diagnosis[diagnosis][:n_per_class]
        
        # Calculate grid size
        n_rows = len(samples)
        n_cols = n_bits
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Binarized Images - Diagnosis {diagnosis}\n(Misclassified Samples)', 
                     fontsize=14, fontweight='bold')
        
        for row_idx, sample in enumerate(samples):
            test_idx = sample['test_idx']
            img = X_test[test_idx].squeeze()
            
            # Binarize the image
            img_normalized = ((img - np.quantile(img, 0.001)) / 
                            (np.quantile(img, 0.999) - np.quantile(img, 0.001))).clip(0, 1)
            thresholds = np.linspace(0, 1-(1/n_bits), n_bits)
            
            for col_idx in range(n_bits):
                ax = axes[row_idx, col_idx]
                binary_layer = (img_normalized > thresholds[col_idx]).astype(int)
                
                ax.imshow(binary_layer, cmap='binary')
                ax.axis('off')
                
                if col_idx == 0:
                    ax.set_ylabel(f"ID:{sample['patient_id']}\n"
                                f"T:{sample['true_label']} P:{sample['predicted_label']}",
                                fontsize=10, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(f'Bit {col_idx+1}', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/binarized_diagnosis_{diagnosis}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for Diagnosis {diagnosis}")

def visualize_binarized_correct(X_test, Y_test, patient_idx, diagnosis,
                                n_bits, save_path, n_per_class=3):
    """Visualize binarized images of correctly classified samples grouped by diagnosis"""
    
    # Group correct samples by diagnosis
    samples_by_diagnosis = defaultdict(list)
    for idx in range(len(X_test)):
        diag = int(diagnosis[idx]) if diagnosis is not None else int(Y_test[idx])
        
        samples_by_diagnosis[diag].append({
            'test_idx': idx,
            'patient_id': patient_idx[idx],
            'diagnosis': diag
        })
    
    # Create visualization for each diagnosis
    diagnoses = sorted(samples_by_diagnosis.keys())
    
    for diagnosis_label in diagnoses:
        samples = samples_by_diagnosis[diagnosis_label][:n_per_class]
        
        n_rows = len(samples)
        n_cols = n_bits
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Binarized Images - Diagnosis {diagnosis_label}\n(Correctly Classified Samples)', 
                     fontsize=14, fontweight='bold')
        
        for row_idx, sample in enumerate(samples):
            test_idx = sample['test_idx']
            img = X_test[test_idx].squeeze()
            
            img_normalized = ((img - np.quantile(img, 0.001)) / 
                            (np.quantile(img, 0.999) - np.quantile(img, 0.001))).clip(0, 1)
            thresholds = np.linspace(0, 1-(1/n_bits), n_bits)
            
            for col_idx in range(n_bits):
                ax = axes[row_idx, col_idx]
                binary_layer = (img_normalized > thresholds[col_idx]).astype(int)
                
                ax.imshow(binary_layer, cmap='binary')
                ax.axis('off')
                
                if col_idx == 0:
                    ax.set_ylabel(f"ID:{sample['patient_id']}",
                                fontsize=10, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(f'Bit {col_idx+1}', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/binarized_correct_diagnosis_{diagnosis_label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correct classification visualization for Diagnosis {diagnosis_label}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Misclassified Samples Analysis')
    parser.add_argument('--exp_name', required=True, type=str, 
                       help='Experiment name')
    parser.add_argument('--n_per_class', default=3, type=int,
                       help='Number of samples to visualize per diagnosis')
    parser.add_argument('--dataset_path', required=True, type=str,
                       help='Path to the dataset H5 file')
    parser.add_argument('--n_bits', default=16, type=int,
                       help='Number of bits used for binarization')
    args = parser.parse_args()
    
    diagnosis_names = {
        0: 'Normal',
        1: 'Diagnosis_1',
        2: 'Diagnosis_2',
        3: 'Diagnosis_3'
    }
    
    # Set up paths
    analysis_path = f'path_to_project_folder/perf/{args.exp_name}/analysis'
    txt_file = f'{analysis_path}/misclassified_details.txt'
    output_path = f'{analysis_path}/misclassified_visualization'
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Analyzing misclassified samples from: {args.exp_name}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} not found!")
    
    # Parse misclassified data
    print("Parsing misclassified details...")
    misclassified_data = analyse_misclassified_details(txt_file)
    print(f"Found {len(misclassified_data)} misclassified samples")
    
    # Load test data
    print("Loading test data...")
    X_test, Y_test, patient_idx, diagnosis = get_test_data(args.dataset_path)
    
    if len(X_test.shape) == 3:
        X_test = X_test[..., np.newaxis]
    
    # Plot distribution
    print("Plotting distribution...")
    plot_misclassified_distribution(misclassified_data, X_test, Y_test, patient_idx,
                                   diagnosis, output_path, diagnosis_names)
    
    # Visualize binarized images
    print("Visualizing binarized images (misclassified)...")
    visualize_binarized_misclassified(misclassified_data, X_test, Y_test, patient_idx,
                                      args.n_bits, output_path, 
                                      n_per_class=args.n_per_class)
    
    # Visualize binarized images of correctly classified
    print("Visualizing binarized images (correctly classified)...")
    visualize_binarized_correct(X_test, Y_test, patient_idx, diagnosis,
                               args.n_bits, output_path, 
                               n_per_class=args.n_per_class)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print(f"All results saved to: {output_path}")

