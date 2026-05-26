import os
import sys
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Prefer SLURM_SUBMIT_DIR when running under sbatch; otherwise use cwd
PROJECT_ROOT = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())

# --- MODEL ARCHITECTURE ---
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout_rate=0.5):
        super(SimpleLSTM, self).__init__()
        # padding_idx=0 ensures the <pad> token doesn't contribute to the gradients
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate) # Adding dropout layer for regularization
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]
        out = self.fc(last_hidden)
        return self.sigmoid(out).squeeze()

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Classical LSTM Training")
    
    # JSON Config File Option
    parser.add_argument('--config_file', type=str, default=None, 
                        help="Path to the JSON config file. Overrides the arguments below.")
    
    # Data Settings
    parser.add_argument('--data_source', type=str, default='bach', choices=['bach', 'sst2', 'synthetic'],
                        help="Data source: 'bach' for local dataset, 'sst2' or 'synthetic' for built-in datasets.")
    parser.add_argument('--num_train', type=int, default=50, help="Number of training samples.")
    parser.add_argument('--num_val', type=int, default=10, help="Number of validation samples.")
    parser.add_argument('--num_test', type=int, default=10, help="Number of test samples.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    # Model and Training Settings
    parser.add_argument('--embed_dim', type=int, default=16, help="Embedding dimension size.")
    parser.add_argument('--hidden_dim', type=int, default=32, help="LSTM hidden layer dimension size.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout Rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 Regularization Weight Decay")
    parser.add_argument('--threshold', type=float, default=0.5, help="Classification Threshold")
    parser.add_argument('--batch_size', type=int, default=16, help="Training and testing batch size.")
    
    # Output Settings
    default_out = os.path.join(PROJECT_ROOT, 'output_classical')
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

    print(f"[*] REAL DATASET SIZES -> Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    if len(val_data) == 0 or len(test_data) == 0:
        print("!!! WARNING: Validation or Test set is empty! Decrease --num_train or get more data !!!")
    return train_data, val_data, test_data

def process_data(train_data, val_data, test_data):
    """Builds vocabulary and pads sentences to the same length."""
    all_sentences = [d[0] for d in train_data + val_data + test_data]
    
    # <pad> is 0, <unk> is 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for sent in all_sentences:
        for word in sent.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
                
    vocab_size = len(vocab)
    max_len = max(len(sent.split()) for sent in all_sentences)
    
    def encode(data):
        encoded, labels = [], []
        for sent, label in data:
            tokens = [vocab.get(w, vocab["<unk>"]) for w in sent.lower().split()]
            encoded.append(tokens + [vocab["<pad>"]] * (max_len - len(tokens)))
            labels.append(label)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

    X_train, y_train = encode(train_data)
    X_val, y_val = encode(val_data)
    X_test, y_test = encode(test_data)
    return X_train, y_train, X_val, y_val, X_test, y_test, vocab_size

def calc_metrics(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

def plot_and_save_metrics(history, best_epoch, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    axs[0].plot(epochs, history['val_loss'], label='Val Loss', color='orange')
    axs[0].axvline(x=best_epoch, color='red', linestyle=':', label='Best Model')
    axs[0].set_title('Loss')
    axs[0].legend()

    axs[1].plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    axs[1].plot(epochs, history['val_acc'], label='Val Acc', color='orange')
    axs[1].axvline(x=best_epoch, color='red', linestyle=':', label='Best Model')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    axs[2].plot(epochs, history['train_f1'], label='Train F1', color='blue')
    axs[2].plot(epochs, history['val_f1'], label='Val F1', color='orange')
    axs[2].axvline(x=best_epoch, color='red', linestyle=':', label='Best Model')
    axs[2].set_title('F1 Score')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

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
    print(f"MODEL                 : CLASSICAL LSTM")
    print(f"DATA SOURCE SELECTION : {args.data_source.upper()}")
    print(f"TRAINING SIZE         : {args.num_train}")
    print(f"VALIDATION SIZE       : {args.num_val}")
    print(f"TEST SIZE             : {args.num_test}")
    print(f"WEIGHT DECAY          : {args.weight_decay}")
    print(f"DROPOUT RATE          : {args.dropout}")
    print(f"CLASSIFICATION THRESHOLD : {args.threshold}")
    print(f"BATCH SIZE            : {args.batch_size}")
    print(f"EMBED/HIDDEN DIMS     : {args.embed_dim} / {args.hidden_dim}")
    print("========================================\n")

    print("--- Preparing Dataset and Tokenization ---")
    train_data, val_data, test_data = get_data(args)
    X_train, y_train, X_val, y_val, X_test, y_test, vocab_size = process_data(train_data, val_data, test_data)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    print("--- Initializing Classical Model ---")
    model = SimpleLSTM(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, dropout_rate=args.dropout)    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    best_val_loss = float('inf')
    best_epoch = 1
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    print(f"\n--- Starting Classical LSTM Training (Epochs: {args.epochs}) ---")
    
    for epoch in range(args.epochs):
        # --- TRAIN PHASE ---
        model.train()
        epoch_loss = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            
            # Handle edge case where batch_size=1 causes squeeze() to remove all dimensions
            if preds.dim() == 0: preds = preds.unsqueeze(0)
                
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds_bin = (preds > 0.5).float()
            tp += ((preds_bin == 1) & (batch_y == 1)).sum().item()
            fp += ((preds_bin == 1) & (batch_y == 0)).sum().item()
            tn += ((preds_bin == 0) & (batch_y == 0)).sum().item()
            fn += ((preds_bin == 0) & (batch_y == 1)).sum().item()
            
        train_loss = epoch_loss / len(train_loader)
        train_acc, _, _, train_f1 = calc_metrics(tp, fp, tn, fn)
        
        # --- VALIDATION PHASE ---
        model.eval() # Dropouts disable
        val_loss = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                if preds.dim() == 0: preds = preds.unsqueeze(0)
                
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                
                preds_bin = (preds > args.threshold).float()
                tp += ((preds_bin == 1) & (batch_y == 1)).sum().item()
                fp += ((preds_bin == 1) & (batch_y == 0)).sum().item()
                tn += ((preds_bin == 0) & (batch_y == 0)).sum().item()
                fn += ((preds_bin == 0) & (batch_y == 1)).sum().item()
                
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc, _, _, val_f1 = calc_metrics(tp, fp, tn, fn)

        # Best Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            mark = "*(Best)*"
        else:
            mark = ""

        print(f"Ep {epoch+1:02d}/{args.epochs} | "
              f"TR Loss: {train_loss:.4f} Acc: {train_acc:.2f} | "
              f"VAL Loss: {val_loss:.4f} Acc: {val_acc:.2f} {mark}", flush=True)

        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1); history['val_f1'].append(val_f1)
            
    plot_and_save_metrics(history, best_epoch, args.output_dir)
    
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # --- TEST SET EVALUATION WITH BEST MODEL ---
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds = model(batch_X)
            if preds.dim() == 0: preds = preds.unsqueeze(0)
            test_preds.extend(preds.tolist())
            test_labels.extend(batch_y.tolist())
            
    thresholds_to_test = sorted(list(set([0.3, 0.4, 0.5, 0.6, 0.7, args.threshold])))
    evaluate_thresholds(test_preds, test_labels, thresholds_to_test)

if __name__ == "__main__":
    main()