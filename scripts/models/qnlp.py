import os
import sys
import json
import random
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lambeq import cups_reader, PennyLaneModel, IQPAnsatz, AtomicType

# Prefer SLURM_SUBMIT_DIR when running under sbatch; otherwise use cwd
PROJECT_ROOT = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Quantum NLP Training")
    
    # JSON Config File Option
    parser.add_argument('--config_file', type=str, default=None, 
                        help="Path to the JSON config file. If provided, overrides the arguments below.")
    
    # Data Settings
    parser.add_argument('--data_source', type=str, default='bach', choices=['bach', 'sst2', 'synthetic'],
                        help="Data source: 'bach' for local Bach dataset, 'sst2' or 'synthetic' for built-in datasets.")
    parser.add_argument('--num_train', type=int, default=50, help="Number of training samples.")
    parser.add_argument('--num_val', type=int, default=10, help="Number of validation samples.")
    parser.add_argument('--num_test', type=int, default=10, help="Number of test samples.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    # Model and Training Settings
    parser.add_argument('--n_layers', type=int, default=2, help="Number of quantum circuit layers (IQP Ansatz).")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 Regularization for Quantum Angles")
    parser.add_argument('--threshold', type=float, default=0.5, help="Classification Threshold")

    # Output Settings
    default_out = os.path.join(PROJECT_ROOT, 'output_qnlp')
    parser.add_argument('--output_dir', type=str, default=default_out, help="Directory to save the outputs.")

    args = parser.parse_args()

    # Override arguments if a config file is provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
            vars(args).update(config_dict)

    return args

def save_config(args):
    """Saves the provided arguments as a JSON file in the output directory."""
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"[*] Configuration saved to: {config_path}")

def get_data(args):
    """Fetches the dataset based on the selected data source."""
    if args.data_source == 'bach':
        # Automatically use the Bach JSON file (located under scripts/data_utils)
        data_path = os.path.join(PROJECT_ROOT, 'scripts', 'data_utils', 'bach_qnlp_sentiment_dataset.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        random.seed(args.seed)
        random.shuffle(data)
        
    else: # sst2 or synthetic
        # Ensure the data_utils folder is on sys.path so imports work under sbatch
        data_utils_path = os.path.join(PROJECT_ROOT, 'scripts', 'data_utils')
        if data_utils_path not in sys.path:
            sys.path.insert(0, data_utils_path)
        from sentence_data_utils import get_dataset
        data, test_data = get_dataset(args.data_source, args.num_train + args.num_val + args.num_test, args.seed)
        data = data + test_data

    train_data = data[:args.num_train]
    val_data = data[args.num_train : args.num_train + args.num_val]
    test_data = data[args.num_train + args.num_val : args.num_train + args.num_val + args.num_test]
    return train_data, val_data, test_data

def calc_metrics(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

def plot_and_save_metrics(history, best_epoch, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Plot
    axs[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    axs[0].plot(epochs, history['val_loss'], label='Val Loss', color='red', linestyle='--')
    axs[0].axvline(x=best_epoch, color='red', linestyle=':', label='Best Model')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Accuracy Plot
    axs[1].plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    axs[1].plot(epochs, history['val_acc'], label='Val Acc', color='red', linestyle='--')
    axs[1].axvline(x=best_epoch, color='red', linestyle=':', label='Best Model')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # F1 Score Plot
    axs[2].plot(epochs, history['train_f1'], label='Train F1', color='blue')
    axs[2].plot(epochs, history['val_f1'], label='Val F1', color='red', linestyle='--')
    axs[2].axvline(x=best_epoch, color='red', linestyle=':', label='Best Model')
    axs[2].set_title('F1 Score over Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('F1 Score')
    axs[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[*] Training plots saved to: {plot_path}")

def evaluate_thresholds(preds, labels, thresholds):
    print("\n" + "="*50)
    print("BEST MODEL TEST SET EVALUATION (MULTIPLE THRESHOLDS)")
    print("="*50)
    print(f"{'Threshold':<10} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 50)
    args = parse_args()
    
    for t in thresholds:
        tp = sum(1 for p, y in zip(preds, labels) if p > t and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p > t and y == 0)
        tn = sum(1 for p, y in zip(preds, labels) if p <= t and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p <= t and y == 1)
        acc, prec, rec, f1 = calc_metrics(tp, fp, tn, fn)
        
        marker = "<-- (Arg)" if t == args.threshold else "" 
        print(f"{t:<10.2f} | {acc:<8.2f} | {prec:<8.2f} | {rec:<8.2f} | {f1:<8.2f} {marker}")
    print("="*50 + "\n")

def main():
    args = parse_args()
    save_config(args)
    
    print("========================================")
    print(f"DATA SOURCE SELECTION : {args.data_source.upper()}")
    print(f"TRAINING SIZE         : {args.num_train}")
    print(f"VALIDATION SIZE       : {args.num_val}")
    print(f"TEST SIZE             : {args.num_test}")
    print(f"WEIGHT DECAY          : {args.weight_decay}")
    print(f"CLASSIFICATION THRESHOLD : {args.threshold}")
    print(f"QUANTUM LAYERS        : {args.n_layers}")
    print("========================================\n")

    print("--- Preparing Dataset ---")
    train_data, val_data, test_data = get_data(args)

    print("Generating DisCoCat grammar diagrams...")
    diagrams = [cups_reader.sentence2diagram(d[0]) for d in train_data + val_data + test_data]
    
    print("Generating quantum circuits (IQP Ansatz)...")
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=args.n_layers)
    circuits = [ansatz(diag) for diag in diagrams]

    train_circuits = circuits[:len(train_data)]
    val_circuits = circuits[len(train_data):len(train_data)+len(val_data)]
    test_circuits = circuits[len(train_data)+len(val_data):]
    
    # Model initialization (Providing both train and test circuits to build the complete vocabulary)
    model = PennyLaneModel.from_diagrams(circuits, probabilities=True)
    model.initialise_weights()
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}    
    
    best_val_loss = float('inf')
    best_epoch = 1
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    print(f"\n--- Starting Quantum Model Training (Epochs: {args.epochs}) ---")
    for epoch in range(args.epochs):
        # --- TRAIN PHASE ---
        epoch_loss = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for i, circuit in enumerate(train_circuits):
            optimizer.zero_grad()
            pred = model([circuit])[0][1] 
            y_true = torch.tensor(train_data[i][1], dtype=torch.float32)
            
            loss = criterion(pred, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            p_label = 1 if pred.item() > 0.5 else 0
            t_label = int(y_true.item())
            if p_label == 1 and t_label == 1: tp += 1
            elif p_label == 1 and t_label == 0: fp += 1
            elif p_label == 0 and t_label == 0: tn += 1
            elif p_label == 0 and t_label == 1: fn += 1
                
        train_loss = epoch_loss / len(train_circuits)
        train_acc, _, _, train_f1 = calc_metrics(tp, fp, tn, fn)

        # --- VALIDATION PHASE ---
        val_loss = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for i, circuit in enumerate(val_circuits):
                pred = model([circuit])[0][1]
                y_true = torch.tensor(val_data[i][1], dtype=torch.float32)
                loss = criterion(pred, y_true)
                val_loss += loss.item()
                
                p_label = 1 if pred.item() > args.threshold else 0
                if p_label == 1 and y_true.item() == 1: tp += 1
                elif p_label == 1 and y_true.item() == 0: fp += 1
                elif p_label == 0 and y_true.item() == 0: tn += 1
                elif p_label == 0 and y_true.item() == 1: fn += 1

        val_loss = val_loss / len(val_circuits) if len(val_circuits) > 0 else 0.0
        val_acc, _, _, val_f1 = calc_metrics(tp, fp, tn, fn)

        # Best Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            mark = "*(Best)*"
        else:
            mark = ""
        
        # Log history
        print(f"Ep {epoch+1:02d}/{args.epochs} | "
              f"TR Loss: {train_loss:.4f} Acc: {train_acc:.2f} | "
              f"VAL Loss: {val_loss:.4f} Acc: {val_acc:.2f} {mark}", flush=True)

        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1); history['val_f1'].append(val_f1)

    # Generate and save plots
    plot_and_save_metrics(history, best_epoch, args.output_dir)
    
    # Save raw history data to JSON
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # --- TEST SET EVALUATION WITH BEST MODEL ---
    model.load_state_dict(torch.load(best_model_path))
    test_preds, test_labels = [], []
    with torch.no_grad():
        for i, circuit in enumerate(test_circuits):
            pred = model([circuit])[0][1].item()
            test_preds.append(pred)
            test_labels.append(test_data[i][1])

    # Evaluate multiple thresholds including the one used during training
    thresholds_to_test = sorted(list(set([0.3, 0.4, 0.5, 0.6, 0.7, args.threshold])))
    evaluate_thresholds(test_preds, test_labels, thresholds_to_test)

if __name__ == "__main__":
    main()