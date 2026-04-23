import os
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time, strftime
import h5py
import argparse
import json
from bz2 import BZ2File
import pickle
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, 
    precision_score, recall_score, accuracy_score, confusion_matrix
)

def binarize_images(X, n_bits):
    n_samples, x_dim, y_dim, _ = X.shape
    vmin = np.quantile(X, 0.001)
    vmax = np.quantile(X, 0.999)
    X =  ((X - vmin) / (vmax - vmin)).clip(0, 1)
    X = X.reshape(n_samples, -1)
    thresholds = np.linspace(0, 1-(1/n_bits), n_bits)
    x_bin = (X[...,np.newaxis] > thresholds).astype(np.uint32)
    return x_bin.reshape(X.shape[0], x_dim, y_dim, n_bits)

def get_train_val_data(ds_name, k_fold_index=4):
    train_val_plans = {
        4: ([0, 1, 2, 3], 4),
        3: ([0, 1, 2, 4], 3),
        2: ([0, 1, 3, 4], 2),
        1: ([0, 2, 3, 4], 1),
        0: ([1, 2, 3, 4], 0),
    }
    train_folds, val_fold = train_val_plans[k_fold_index]
    with h5py.File(ds_name, 'r') as f:
        X_train = []
        Y_train = []
        for fold in train_folds:
            X_train.append(f[f'fold_{fold}']['image'][:])
            Y_train.append(f[f'fold_{fold}']['target'][:])
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)
        X_val = f[f'fold_{val_fold}']['image'][:]
        Y_val = f[f'fold_{val_fold}']['target'][:]
    return (X_train, Y_train), (X_val, Y_val)

def get_test_data(ds_name):
    with h5py.File(ds_name, 'r') as f:
        X_test = f['fold_5']['image'][:]
        Y_test = f['fold_5']['target'][:]
    return X_test, Y_test

def calculate_metrics(y_true, y_pred):
    """Calculate all performance metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    
    # AUC only for binary classification
    if len(np.unique(y_true)) == 2:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics

def save_checkpoint(exp_name, epoch, tm, best_val_mcc):
    """Save training checkpoint"""
    checkpoint_path = f'{project_dir}/perf/{exp_name}/checkpoint_epoch_{epoch}.pkl'
    state_dict = tm.save()
    checkpoint = {
        'epoch': epoch,
        'model_state': state_dict,
        'best_val_mcc': best_val_mcc
    }
    with BZ2File(checkpoint_path, "wb") as file:
        pickle.dump(checkpoint, file)
    
    # Save latest checkpoint marker
    with open(f'{project_dir}/perf/{exp_name}/latest_checkpoint.txt', 'w') as f:
        f.write(str(epoch))
    
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(exp_name):
    """Load the latest checkpoint"""
    try:
        with open(f'{project_dir}/perf/{exp_name}/latest_checkpoint.txt', 'r') as f:
            last_epoch = int(f.read().strip())
        
        checkpoint_path = f'{project_dir}/perf/{exp_name}/checkpoint_epoch_{last_epoch}.pkl'
        with BZ2File(checkpoint_path, "rb") as file:
            checkpoint = pickle.load(file)
        
        print(f"Resuming from epoch {checkpoint['epoch']}")
        return checkpoint
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")
        return None

# Fixed hyperparameters
number_of_clauses = 4000
T = 10000
s = 1.4
patch_dim_size = 25
n_bits = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, type=str, help="Experiment name (required for resume)")
    parser.add_argument("--resume", action='store_true', help="Resume from checkpoint")
    parser.add_argument("--epochs", default=50, type=int, help="Epochs for final training")
    parser.add_argument("--checkpoint_freq", default=5, type=int, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    
    exp_name = args.exp_name
    print(f"Experiment Name: {exp_name}")
    project_dir = 'path_to_your_project'                   # Update with actual project directory
    os.makedirs(f'{project_dir}/perf/{exp_name}', exist_ok=True)

    dataset_path = 'path_to_your_dataset.h5'                # Update with actual dataset path


    # Load full training data for final model training
    print("\nLoading full training data for final model...")
    (X_train, Y_train), (X_val, Y_val) = get_train_val_data(dataset_path, k_fold_index=4)

    if len(X_train.shape) == 3:
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        
    _, x_dim, y_dim, _ = X_train.shape

    # Use the best n_bits found by Optuna
    X_train_bin = binarize_images(X_train, n_bits).reshape(X_train.shape[0], -1).astype(np.uint32)
    X_val_bin = binarize_images(X_val, n_bits).reshape(X_val.shape[0], -1).astype(np.uint32)

    # Train model with fixed hyperparameters
    print("\n=== Training ===")
    configs = {
        'number_of_clauses': number_of_clauses,
        'T': T,
        's': s,
        'dim': (x_dim, y_dim, n_bits),
        'patch_dim': (patch_dim_size, patch_dim_size)
    }
    
    with open(f'{project_dir}/perf/{exp_name}/best_configs.json', 'w') as f:
        info = {'dataset_path': dataset_path, 'n_bits': n_bits}
        info.update(configs)
        json.dump(info, f, indent=4)

    # Check for checkpoint
    checkpoint = None
    start_epoch = 0
    best_val_mcc = -1.0
    
    if args.resume:
        checkpoint = load_checkpoint(exp_name)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_mcc = checkpoint['best_val_mcc']

    tm = MultiClassConvolutionalTsetlinMachine2D(**configs)
    
    # Load model state if resuming
    if checkpoint:
        tm.load(checkpoint['model_state'])
        print(f"Model state restored from epoch {checkpoint['epoch']}")

    # Create or append to log file
    if start_epoch == 0:
        with open(f'{project_dir}/perf/{exp_name}/training_log.txt', 'w') as log_file:
            log_file.write("Epoch | Train_Acc | Train_F1 | Train_MCC | Train_AUC | Train_Precision | Train_Recall | Val_Acc | Val_F1 | Val_MCC | Val_AUC | Val_Precision | Val_Recall\n")
            log_file.write("-" * 160 + "\n")

    for i in range(start_epoch, args.epochs):
        batch_size = 1024
        n_samples = X_train_bin.shape[0]
        seg = (i % ((n_samples + batch_size - 1) // batch_size)) * batch_size
        end_seg = min(seg + batch_size, n_samples)
        
        X_train_bin_seg = X_train_bin[seg:end_seg]
        Y_train_seg = Y_train[seg:end_seg]
        
        print(f"\nEpoch {i + 1}\n-----------------")
        start_training = time()
        tm.fit(X_train_bin_seg, Y_train_seg, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        y_val_pred = tm.predict(X_val_bin)
        val_metrics = calculate_metrics(Y_val, y_val_pred)
        stop_testing = time()

        y_train_pred = tm.predict(X_train_bin_seg)
        train_metrics = calculate_metrics(Y_train_seg, y_train_pred)

        print(f"Epoch {i + 1} | Train Time: {stop_training - start_training:.2f}s, Val Time: {stop_testing - start_testing:.2f}s")
        print(f"Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, MCC: {train_metrics['mcc']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Train - Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, MCC: {val_metrics['mcc']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        with open(f'{project_dir}/perf/{exp_name}/training_log.txt', 'a') as log_file:
            log_file.write(f"{i+1:02d} | {train_metrics['accuracy']:.4f} | {train_metrics['f1']:.4f} | {train_metrics['mcc']:.4f} | ")
            log_file.write(f"{train_metrics['auc']:.4f} | {train_metrics['precision']:.4f} | {train_metrics['recall']:.4f} | ")
            log_file.write(f"{val_metrics['accuracy']:.4f} | {val_metrics['f1']:.4f} | {val_metrics['mcc']:.4f} | {val_metrics['auc']:.4f} | ")
            log_file.write(f"{val_metrics['precision']:.4f} | {val_metrics['recall']:.4f}\n")

        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            state_dict = tm.save()
            with BZ2File(f'{project_dir}/perf/{exp_name}/best_model.tm', "wb") as file:
                pickle.dump(state_dict, file)
            with open(f'{project_dir}/perf/{exp_name}/best_model.txt', 'w') as log_file:
                log_file.write(f"Epoch {i + 1:02d}\n")
                log_file.write(f"Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, MCC: {val_metrics['mcc']:.4f}\n")
                log_file.write(f"AUC: {val_metrics['auc']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}\n")

        # Save checkpoint periodically
        if (i + 1) % args.checkpoint_freq == 0:
            save_checkpoint(exp_name, i + 1, tm, best_val_mcc)

    print("Training completed.")

    # Test final model
    with BZ2File(f'{project_dir}/perf/{exp_name}/best_model.tm', "rb") as file:
        loaded_dict = pickle.load(file)

    new_tm = MultiClassConvolutionalTsetlinMachine2D(**configs)
    new_tm.load(loaded_dict)
    
    X_test, Y_test = get_test_data(dataset_path)
    if len(X_test.shape) == 3:
        X_test = X_test[..., np.newaxis]
    X_test_bin = binarize_images(X_test, n_bits).reshape(X_test.shape[0], -1).astype(np.uint32)
    
    start_testing = time()
    y_test_pred = new_tm.predict(X_test_bin)
    test_metrics = calculate_metrics(Y_test, y_test_pred)
    stop_testing = time()
    
    print(f"\n=== Test Results ===")
    print(f"Test Time: {stop_testing - start_testing:.2f}s")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    
    with open(f'{project_dir}/perf/{exp_name}/test_result.txt', 'w') as log_file:
        log_file.write(f"Test Time: {stop_testing - start_testing:.2f}s\n")
        log_file.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        log_file.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
        log_file.write(f"MCC: {test_metrics['mcc']:.4f}\n")
        log_file.write(f"AUC: {test_metrics['auc']:.4f}\n")
        log_file.write(f"Precision: {test_metrics['precision']:.4f}\n")
        log_file.write(f"Recall: {test_metrics['recall']:.4f}\n")
        log_file.write(f"\nConfusion Matrix:\n{confusion_matrix(Y_test, y_test_pred)}\n")