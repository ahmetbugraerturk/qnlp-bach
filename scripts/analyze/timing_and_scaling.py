# timing_and_scaling.py
# =====================
# Measures wall-clock time per training step for LSTM and QNLP
# across vocabulary sizes and sentence lengths.
# Run this from your project directory:
#   python timing_and_scaling.py
# It will print a CSV table and save timing_results.csv
# Upload timing_results.csv back and I will generate the plots.

import time, csv, random, torch, torch.nn as nn, torch.optim as optim

# ── LSTM timing ────────────────────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        e = self.embedding(x)
        _, (h, _) = self.lstm(e)
        return self.sig(self.fc(h[-1])).squeeze()

def time_lstm(vocab_size, seq_len, n_steps=20):
    model = SimpleLSTM(vocab_size)
    opt   = optim.Adam(model.parameters(), lr=0.01)
    crit  = nn.BCELoss()
    times = []
    for _ in range(n_steps):
        x = torch.randint(1, vocab_size, (8, seq_len))
        y = torch.randint(0, 2, (8,)).float()
        t0 = time.perf_counter()
        opt.zero_grad()
        p = model(x)
        if p.dim() == 0: p = p.unsqueeze(0)
        loss = crit(p, y)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    return sum(times[5:]) / len(times[5:])   # skip warm-up

# ── QNLP timing ───────────────────────────────────────────────────────────
def time_qnlp(n_tokens, n_layers, n_steps=10):
    """Returns avg seconds per forward+backward pass for one sample."""
    try:
        from lambeq import cups_reader, IQPAnsatz, PennyLaneModel, AtomicType
        words    = [f"w{i}" for i in range(n_tokens)]
        sentence = " ".join(words)
        diagram  = cups_reader.sentence2diagram(sentence)
        ansatz   = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                              n_layers=n_layers)
        circuit  = ansatz(diagram)
        model    = PennyLaneModel.from_diagrams([circuit], probabilities=True)
        model.initialise_weights()
        opt  = torch.optim.Adam(model.parameters(), lr=0.1)
        crit = nn.BCELoss()
        times = []
        for step in range(n_steps):
            y = torch.tensor(float(step % 2))
            t0 = time.perf_counter()
            opt.zero_grad()
            pred = model([circuit])[0][1]
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            times.append(time.perf_counter() - t0)
        return sum(times[2:]) / len(times[2:])
    except Exception as e:
        print(f"  QNLP error (n_tokens={n_tokens}, n_layers={n_layers}): {e}")
        return None

# ── Configs ────────────────────────────────────────────────────────────────
LSTM_CONFIGS = [
    # (vocab_size, seq_len)
    (50,  3), (100, 3), (200, 3), (500, 3),
    (50,  5), (100, 5), (200, 5),
]
QNLP_CONFIGS = [
    # (n_tokens, n_layers)
    (2, 1), (2, 2),
    (3, 1), (3, 2),
    (4, 1), (4, 2),
    (5, 1), (5, 2),
    (6, 1), (6, 2),
]

results = []

print("=== LSTM TIMING ===")
for vocab_size, seq_len in LSTM_CONFIGS:
    t = time_lstm(vocab_size, seq_len)
    n_params = vocab_size*16 + 4*(16*32 + 32*32 + 32) + 32 + 1
    row = {"model": "LSTM", "vocab_size": vocab_size, "seq_len": seq_len,
           "n_tokens": seq_len, "n_layers": "-", "n_params": n_params,
           "n_qubits": "-", "time_per_step_s": round(t, 6)}
    results.append(row)
    print(f"  vocab={vocab_size:4d} seq={seq_len} | {t*1000:.3f} ms | params={n_params}")

print("\n=== QNLP TIMING ===")
for n_tokens, n_layers in QNLP_CONFIGS:
    t = time_qnlp(n_tokens, n_layers)
    if t is None:
        continue
    n_qubits = 2*n_tokens + 1
    n_params  = 3 + n_tokens * 2 * n_layers   # START(3 fixed) + word params
    row = {"model": "QNLP", "vocab_size": "-", "seq_len": n_tokens,
           "n_tokens": n_tokens, "n_layers": n_layers, "n_params": n_params,
           "n_qubits": n_qubits, "time_per_step_s": round(t, 6)}
    results.append(row)
    print(f"  tokens={n_tokens} layers={n_layers} | {t*1000:.3f} ms | "
          f"qubits={n_qubits} params={n_params}")

# ── Save CSV ───────────────────────────────────────────────────────────────
with open("timing_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\nSaved: timing_results.csv — upload this file back.")