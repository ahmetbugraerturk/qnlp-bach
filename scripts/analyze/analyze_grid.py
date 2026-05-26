import os
import sys
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Make project root available (prefer SLURM_SUBMIT_DIR when running under sbatch)
PROJECT_ROOT = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())
# Allow importing helper modules from the project root
sys.path.append(PROJECT_ROOT)

def calculate_baselines(data_source, num_train, num_val, num_test, seed):
    """
    Loads data using the same logic as the training scripts.
    Computes majority-class and random baselines based on the test set label distribution.
    """
    try:
        total_samples = num_train + num_val + num_test
        
        if data_source == 'bach':
            data_path = os.path.join(PROJECT_ROOT, 'scripts', 'data_utils', 'bach_qnlp_sentiment_dataset.json')
            with open(data_path, 'r') as f:
                raw_data = json.load(f)
            # Apply a length filter to avoid circuit explosion / OOM
            data = [d for d in raw_data if len(d[0].split()) <= 6]
            random.seed(seed)
            random.shuffle(data)
        else:
            # Make sure the local data_utils module is importable
            data_utils_path = os.path.join(PROJECT_ROOT, 'scripts', 'data_utils')
            if data_utils_path not in sys.path:
                sys.path.insert(0, data_utils_path)
            from sentence_data_utils import get_dataset
            data, _ = get_dataset(data_source, num_samples=total_samples, seed=seed)
            
        test_data = data[num_train + num_val : total_samples]
        
        if not test_data:
            return 0.5, 0.5 # Fallback in error cases
            
        labels = [d[1] for d in test_data]
        pos_ratio = sum(labels) / len(labels)
        
    # Majority baseline: accuracy if always predicting the majority class
        maj_acc = max(pos_ratio, 1.0 - pos_ratio)
        
    # Random baseline F1 (if predicting randomly). Approximate via expected precision/recall.
        rand_f1 = pos_ratio / (pos_ratio + 0.5) if (pos_ratio + 0.5) > 0 else 0.0
        
        return maj_acc, rand_f1
        
    except Exception as e:
        print(f"[WARNING] Could not compute baselines for ({data_source}): {e}")
        return 0.5, 0.5


def parse_test_metrics_from_log(log_path):
    """Extract best Threshold, Test F1 and Test Acc values from the log file."""
    best_test_f1, best_test_acc, optimal_thresh = 0.0, 0.0, 0.0
    if not os.path.exists(log_path): return best_test_f1, best_test_acc, optimal_thresh
        
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        in_table = False
        for line in lines:
            if "Threshold" in line and "Acc" in line and "F1" in line:
                in_table = True; continue
            if in_table and ("---" in line or "===" in line):
                if "===" in line: in_table = False
                continue
                
            if in_table:
                parts = line.split('|')
                if len(parts) >= 5:
                    try:
                        thresh = float(parts[0].strip())
                        acc = float(parts[1].strip())
                        f1 = float(parts[4].split('<')[0].strip())
                        
                        if f1 > best_test_f1:
                            best_test_f1, best_test_acc, optimal_thresh = f1, acc, thresh
                    except ValueError: pass
    except Exception: pass
    return best_test_f1, best_test_acc, optimal_thresh

def gather_data(base_dir, model_label):
    records = []
    if not os.path.exists(base_dir): return records
        
    for root, dirs, files in os.walk(base_dir):
        if 'config.json' in files and 'history.json' in files:
            try:
                with open(os.path.join(root, 'config.json'), 'r') as f: config = json.load(f)
                with open(os.path.join(root, 'history.json'), 'r') as f: history = json.load(f)
                
                val_losses = history.get('val_loss', [])
                if not val_losses: continue
                
                best_epoch_idx = int(np.argmin(val_losses))
                
                # Metrics at Best Epoch
                val_loss = val_losses[best_epoch_idx]
                val_f1 = history.get('val_f1', [0])[best_epoch_idx]
                val_acc = history.get('val_acc', [0])[best_epoch_idx]
                train_f1 = history.get('train_f1', [0])[best_epoch_idx]
                overfit_gap = train_f1 - val_f1
                
                # Test Metrics from Log
                log_file = [f for f in files if f.endswith('.log')]
                test_f1, test_acc, opt_thresh = 0.0, 0.0, 0.0
                if log_file:
                    test_f1, test_acc, opt_thresh = parse_test_metrics_from_log(os.path.join(root, log_file[0]))
                
                # Baselines (recompute dataset split to estimate baselines for that seed)
                maj_acc, rand_f1 = calculate_baselines(
                    data_source=config.get('data_source'),
                    num_train=config.get('num_train'),
                    num_val=config.get('num_val', 50),
                    num_test=config.get('num_test', 100),
                    seed=config.get('seed')
                )
                
                record = {
                    'Model': model_label,
                    'Dataset': config.get('data_source', 'unknown'),
                    'Train_Size': config.get('num_train', 0),
                    'Seed': config.get('seed', 0),
                    'Best_Epoch': best_epoch_idx + 1,
                    'Train_F1': train_f1,
                    'Val_Loss': val_loss,
                    'Val_Acc': val_acc,
                    'Val_F1': val_f1,
                    'Overfit_Gap': overfit_gap,
                    'Test_Acc': test_acc,
                    'Test_F1': test_f1,
                    'Maj_Baseline_Acc': maj_acc,
                    'Rand_Baseline_F1': rand_f1
                }
                records.append(record)
            except Exception as e:
                print(f"[ERROR] Failed to read {root}: {e}")
    return records

def analyze_and_plot():
    # Create output directory
    OUT_DIR = "grid_analysis_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"--- Outputs will be saved to: '{OUT_DIR}' ---")
    
    lstm_dir = os.path.join(PROJECT_ROOT, 'output_lstm') if os.path.exists(os.path.join(PROJECT_ROOT, 'output_lstm')) else os.path.join(PROJECT_ROOT, 'output_classical')
    all_records = gather_data(os.path.join(PROJECT_ROOT, 'output_qnlp'), 'QNLP') + gather_data(lstm_dir, 'LSTM')
    
    if not all_records:
        print("No results found! Check the log directories.")
        return
        
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(OUT_DIR, "all_raw_results.csv"), index=False)
    
    # Select best model (by validation loss)
    idx = df.groupby(['Dataset', 'Model', 'Train_Size', 'Seed'])['Val_Loss'].idxmin()
    df_best = df.loc[idx]
    
    # Statistical grouping (mean and std)
    stats_df = df_best.groupby(['Dataset', 'Model', 'Train_Size']).agg(
        Val_F1_Mean=('Val_F1', 'mean'), Val_F1_Std=('Val_F1', 'std'),
        Test_F1_Mean=('Test_F1', 'mean'), Test_F1_Std=('Test_F1', 'std'),
        Val_Acc_Mean=('Val_Acc', 'mean'), Test_Acc_Mean=('Test_Acc', 'mean'),
        Avg_Overfit_Gap=('Overfit_Gap', 'mean'), Avg_Best_Epoch=('Best_Epoch', 'mean'),
        Maj_Baseline_Acc=('Maj_Baseline_Acc', 'mean'), # Average baseline across seeds
        Rand_Baseline_F1=('Rand_Baseline_F1', 'mean')
    ).reset_index()
    
    stats_df.to_csv(os.path.join(OUT_DIR, "aggregated_stats.csv"), index=False)
    print(stats_df.to_string(index=False, float_format="%.4f"))
    
    datasets = stats_df['Dataset'].unique()
    
    # 2x3 Comprehensive visualization
    for ds in datasets:
        ds_data = stats_df[stats_df['Dataset'] == ds]
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Comprehensive Analysis: {ds.upper()}', fontsize=20, fontweight='bold', y=0.96)
        
        x_sizes = sorted(ds_data['Train_Size'].unique())
        
    # Reference points for plotting baselines
        base_f1 = ds_data.groupby('Train_Size')['Rand_Baseline_F1'].mean()
        base_acc = ds_data.groupby('Train_Size')['Maj_Baseline_Acc'].mean()

        for model_name, color, marker in zip(['QNLP', 'LSTM'], ['blue', 'red'], ['o', 's']):
            model_data = ds_data[ds_data['Model'] == model_name].sort_values(by='Train_Size')
            if model_data.empty: continue
            x = model_data['Train_Size']
            
            # [0,0] Test F1 (Generalization)
            axs[0, 0].errorbar(x, model_data['Test_F1_Mean'], yerr=model_data['Test_F1_Std'], fmt=f'-{marker}', color=color, label=model_name, capsize=4)
            axs[0, 0].set_title("1. Test F1 Score")
            axs[0, 0].set_ylabel("F1 Score")
            
            # [0,1] Test Accuracy
            axs[0, 1].plot(x, model_data['Test_Acc_Mean'], f'-{marker}', color=color, label=model_name)
            axs[0, 1].set_title("2. Test Accuracy")
            axs[0, 1].set_ylabel("Accuracy")
            
            # [0,2] Convergence Speed
            axs[0, 2].plot(x, model_data['Avg_Best_Epoch'], f'-{marker}', color=color, label=model_name)
            axs[0, 2].set_title("3. Convergence Speed")
            axs[0, 2].set_ylabel("Epoch (Min Val Loss)")
            
            # [1,0] Validation F1
            axs[1, 0].errorbar(x, model_data['Val_F1_Mean'], yerr=model_data['Val_F1_Std'], fmt=f'-{marker}', color=color, label=model_name, capsize=4)
            axs[1, 0].set_title("4. Validation F1 Score")
            axs[1, 0].set_ylabel("F1 Score")
            
            # [1,1] Validation Accuracy
            axs[1, 1].plot(x, model_data['Val_Acc_Mean'], f'-{marker}', color=color, label=model_name)
            axs[1, 1].set_title("5. Validation Accuracy")
            axs[1, 1].set_ylabel("Accuracy")
            
            # [1,2] Overfit Gap
            axs[1, 2].plot(x, model_data['Avg_Overfit_Gap'], f'-{marker}', color=color, label=model_name)
            axs[1, 2].set_title("6. Overfitting Gap (Train F1 - Val F1)")
            axs[1, 2].set_ylabel("Gap")

    # Add baselines to plots (black dashed lines)
    axs[0, 0].plot(base_f1.index, base_f1.values, 'k--', label="Random Baseline", linewidth=2)
    axs[1, 0].plot(base_f1.index, base_f1.values, 'k--', label="Random Baseline", linewidth=2)
        
    axs[0, 1].plot(base_acc.index, base_acc.values, 'k--', label="Majority Class Baseline", linewidth=2)
    axs[1, 1].plot(base_acc.index, base_acc.values, 'k--', label="Majority Class Baseline", linewidth=2)

    for ax in axs.flat:
        ax.set_xlabel("Training Set Size")
        ax.set_xticks(x_sizes)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plot_path = os.path.join(OUT_DIR, f"detailed_analysis_{ds}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[*] Figure saved: {plot_path}")

if __name__ == "__main__":
    analyze_and_plot()