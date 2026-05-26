"""
generate_plots.py
=================
Generates all figures needed for the QNLP project report.

Usage:
    python generate_plots.py \
        --lstm_dir  output_classical \
        --qnlp_dir  output_qnlp \
        --out_dir   report_figures

The script expects output_classical/ and output_qnlp/ to contain
subdirectories (one per run) each with:
    config.json    — hyperparameters (data_source, num_train, seed, ...)
    history.json   — per-epoch {train_loss, val_loss, train_acc, val_acc,
                                train_f1, val_f1}

Output files (copy all into your Overleaf project folder):
    scaling_analysis.png
    loss_bach_n50.png
    f1_vs_trainsize_bach.png
    f1_vs_trainsize_sst2.png
    f1_vs_trainsize_synthetic.png
    overfit_gap_comparison.png
    variance_comparison_n10.png
"""

import os, json, argparse, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Colours & style ──────────────────────────────────────────────────────────
LSTM_C   = '#C0392B'   # deep red
Q1_C     = '#2471A3'   # steel blue  (1 layer)
Q2_C     = '#1A5276'   # dark navy   (2 layers)
QNLP_C   = '#2471A3'   # general QNLP
RAND_C   = '#7F8C8D'   # grey dashed baseline

plt.rcParams.update({
    'font.family'        : 'serif',
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.grid'          : True,
    'grid.alpha'         : 0.25,
    'grid.linestyle'     : '--',
    'figure.facecolor'   : 'white',
    'axes.facecolor'     : 'white',
    'savefig.facecolor'  : 'white',
})

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

import os
import json

PROJECT_ROOT = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())

def load_runs(base_dir):
    """
    Only inspects direct subfolders of base_dir that start with 'job'.
    Does not recurse into deeper subdirectories.
    """
    runs = []
    
    # Only look at direct entries inside base_dir
    for entry in os.scandir(base_dir):
    # If it's a directory and its name starts with 'job':
        if entry.is_dir() and entry.name.startswith('job'):
            config_path = os.path.join(entry.path, 'config.json')
            history_path = os.path.join(entry.path, 'history.json')
            
            # Check that both expected files are present
            if os.path.exists(config_path) and os.path.exists(history_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                with open(history_path) as f:
                    hist = json.load(f)
                
                runs.append({'cfg': cfg, 'hist': hist, 'path': entry.path})
                
    print(f"  Loaded {len(runs)} runs from {base_dir}")
    return runs


def group_runs(runs, key_fields):
    """Group runs by a tuple of config fields."""
    groups = defaultdict(list)
    for r in runs:
        key = tuple(r['cfg'].get(f, None) for f in key_fields)
        groups[key].append(r)
    return groups


def best_val_f1(hist):
    """Return the best validation F1 seen in a run."""
    return max(hist.get('val_f1', [0]))


def best_test_f1_from_hist(hist):
    """
    We don't store test F1 per epoch in history.json —
    return the val F1 at best val loss epoch as proxy,
    or the last val_f1 if val_loss is unavailable.
    """
    if 'val_loss' in hist and hist['val_loss']:
        best_ep = int(np.argmin(hist['val_loss']))
        return hist['val_f1'][best_ep]
    return max(hist.get('val_f1', [0]))


def aggregate(runs, metric='val_f1', use_best=True):
    """Mean ± std of a metric across runs (each run = one seed)."""
    vals = []
    for r in runs:
        h = r['hist']
        if use_best:
            vals.append(max(h.get(metric, [0])))
        else:
            vals.append(h.get(metric, [0])[-1])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def overfit_gap(runs):
    """
    Mean overfit gap across runs.
    Overfit gap = val_loss_at_best_epoch - train_loss_at_best_epoch
    """
    gaps = []
    for r in runs:
        h = r['hist']
        vl = h.get('val_loss', [])
        tl = h.get('train_loss', [])
        if vl and tl:
            best_ep = int(np.argmin(vl))
            gaps.append(vl[best_ep] - tl[best_ep])
    return float(np.mean(gaps)) if gaps else 0.0


def savefig(fig, path, out_dir):
    out_dir = os.path.join(PROJECT_ROOT, out_dir) if not os.path.isabs(out_dir) else out_dir
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, path)
    fig.savefig(full, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {full}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Scaling analysis  (uses hardcoded timing data from timing_results.csv)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scaling(out_dir):
    # Timing data from timing_and_scaling.py output
    lstm_vocab  = [50,  100, 200, 500]
    lstm_time   = [0.001168, 0.001170, 0.001179, 0.001202]

    qnlp_tokens  = [2,       3,       4,       5,       6]
    qnlp_time_1L = [0.008058, 0.010619, 0.013534, 0.017144, 0.024251]
    qnlp_time_2L = [0.010093, 0.014257, 0.018703, 0.023870, 0.034107]
    qnlp_params_1L = [7,  9,  11, 13, 15]
    qnlp_params_2L = [11, 15, 19, 23, 27]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38)

    # (a) Time vs tokens  (log-log)
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(qnlp_tokens, qnlp_time_1L, 'o-', color=Q1_C, lw=2, ms=7,
              label='QNLP (1 layer)')
    ax.loglog(qnlp_tokens, qnlp_time_2L, 's--', color=Q2_C, lw=2, ms=7,
              label='QNLP (2 layers)')
    ax.axhline(np.mean(lstm_time), color=LSTM_C, lw=2, linestyle=':',
               label=f'LSTM ≈ {np.mean(lstm_time)*1000:.2f} ms (flat)')
    ax.set_xlabel('Sentence length (tokens)')
    ax.set_ylabel('Time per step (s)')
    ax.set_title('(a) Per-step time vs sentence length', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)

    # (b) LSTM time vs vocab  (semilogx)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(lstm_vocab, [t*1000 for t in lstm_time], 'D-',
                 color=LSTM_C, lw=2, ms=7)
    ax2.set_xlabel('Vocabulary size $|\\mathcal{V}|$')
    ax2.set_ylabel('Time per step (ms)')
    ax2.set_title('(b) LSTM time vs vocabulary size', fontweight='bold', fontsize=10)

    # (c) Parameter count  (log scale)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(qnlp_tokens, qnlp_params_1L, 'o-', color=Q1_C, lw=2, ms=7,
             label='QNLP (1 layer)')
    ax3.plot(qnlp_tokens, qnlp_params_2L, 's--', color=Q2_C, lw=2, ms=7,
             label='QNLP (2 layers)')
    # LSTM flat reference
    for vocab, params, ls in [(50, 7105, ':'), (500, 14305, '-.')]:
        ax3.axhline(params, color=LSTM_C, lw=1.5, linestyle=ls,
                    label=f'LSTM ($|\\mathcal{{V}}|$={vocab}) = {params}')
    ax3.set_yscale('log')
    ax3.set_xlabel('Sentence length (tokens)')
    ax3.set_ylabel('Trainable parameters')
    ax3.set_title('(c) Parameter count vs sentence length', fontweight='bold', fontsize=10)
    ax3.legend(fontsize=7)

    # (d) Qubits and classical sim states
    ax4  = fig.add_subplot(gs[1, 1])
    toks = list(range(2, 11))
    ax4.plot(toks, [2*k+1 for k in toks], 'o-', color=Q1_C, lw=2, ms=7,
             label='Physical qubits $2k+1$')
    ax4r = ax4.twinx()
    ax4r.semilogy(toks, [2**(2*k+1) for k in toks], 'v--',
                  color='#884EA0', lw=1.5, ms=6, alpha=0.8,
                  label='Sim states $2^{2k+1}$')
    ax4r.set_ylabel('Classical simulation states', color='#884EA0', fontsize=9)
    ax4r.tick_params(axis='y', colors='#884EA0')
    ax4r.spines['right'].set_visible(True)
    ax4r.spines['right'].set_color('#884EA0')
    ax4.set_xlabel('Sentence length $k$ (tokens)')
    ax4.set_ylabel('Physical qubits', color=Q1_C)
    ax4.tick_params(axis='y', colors=Q1_C)
    ax4.set_title('(d) Qubit count vs sentence length', fontweight='bold', fontsize=10)
    ax4.axvline(5, color='gray', lw=1, linestyle=':', alpha=0.7)
    ax4.text(5.1, 5.5, 'Sim limit\n($k>5$)', fontsize=7.5, color='gray')
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4r.get_legend_handles_labels()
    ax4.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper left')

    fig.suptitle('Empirical Scaling Analysis: LSTM vs QNLP (IQP Ansatz)',
                 fontsize=12, fontweight='bold', y=1.01)
    savefig(fig, 'scaling_analysis.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Loss / Acc / F1 curves for a specific (dataset, N, seed)
#           Best use: Bach, N=50, to show the overfitting story
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(lstm_runs, qnlp_runs, dataset, n_train,
                         seed, out_dir, filename=None):
    """
    Find the run matching (dataset, n_train, seed) for each model
    and plot loss / accuracy / F1 side by side.
    """
    def find_run(runs, dataset, n_train, seed):
        for r in runs:
            c = r['cfg']
            if (c.get('data_source') == dataset and
                    int(c.get('num_train', 0)) == n_train and
                    int(c.get('seed', -1)) == seed):
                return r
        return None

    lr = find_run(lstm_runs, dataset, n_train, seed)
    qr = find_run(qnlp_runs, dataset, n_train, seed)

    if lr is None and qr is None:
        print(f"  [SKIP] No runs found for dataset={dataset} N={n_train} seed={seed}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [('train_loss', 'val_loss', 'Loss', 'lower right'),
               ('train_acc',  'val_acc',  'Accuracy', 'lower right'),
               ('train_f1',   'val_f1',   'F1 Score', 'lower right')]

    for ax, (tr_key, va_key, ylabel, leg_loc) in zip(axes, metrics):
        if lr is not None:
            h = lr['hist']
            ep = range(1, len(h[tr_key])+1)
            ax.plot(ep, h[tr_key], color=LSTM_C, lw=1.8, alpha=0.8,
                    label='LSTM train')
            ax.plot(ep, h[va_key], color=LSTM_C, lw=1.8, ls='--',
                    label='LSTM val')
        if qr is not None:
            h = qr['hist']
            ep = range(1, len(h[tr_key])+1)
            ax.plot(ep, h[tr_key], color=QNLP_C, lw=1.8, alpha=0.8,
                    label='QNLP train')
            ax.plot(ep, h[va_key], color=QNLP_C, lw=1.8, ls='--',
                    label='QNLP val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc=leg_loc)

    fig.suptitle(f'Training curves — {dataset.upper()}  '
                 f'$N_{{train}}={n_train}$  seed={seed}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = filename or f'curves_{dataset}_n{n_train}_seed{seed}.png'
    savefig(fig, fname, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Best val F1 vs training size  (one plot per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_f1_vs_trainsize(lstm_runs, qnlp_runs, dataset, out_dir,
                         train_sizes=None):
    if train_sizes is None:
        train_sizes = [10, 50, 100, 250, 500]

    def get_mean_std(runs, ds, sizes):
        means, stds = [], []
        for n in sizes:
            matching = [r for r in runs
                        if r['cfg'].get('data_source') == ds
                        and int(r['cfg'].get('num_train', 0)) == n]
            if matching:
                m, s = aggregate(matching, metric='val_f1', use_best=True)
            else:
                m, s = None, None
            means.append(m)
            stds.append(s)
        return means, stds

    lstm_m, lstm_s = get_mean_std(lstm_runs, dataset, train_sizes)
    qnlp_m, qnlp_s = get_mean_std(qnlp_runs, dataset, train_sizes)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    def plot_line(sizes, means, stds, color, label, marker):
        xs, ys, es = [], [], []
        for x, m, s in zip(sizes, means, stds):
            if m is not None:
                xs.append(x); ys.append(m); es.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=color, marker=marker,
                        lw=2, ms=7, capsize=4, label=label)

    plot_line(train_sizes, lstm_m, lstm_s, LSTM_C,  'LSTM',  'D')
    plot_line(train_sizes, qnlp_m, qnlp_s, QNLP_C, 'QNLP', 'o')

    ax.set_xlabel('Training size $N_{train}$')
    ax.set_ylabel('Best validation F1 (mean ± std)')
    ax.set_title(f'Validation F1 vs Training Size — {dataset.upper()}',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    plt.tight_layout()
    savefig(fig, f'f1_vs_trainsize_{dataset}.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Overfit gap comparison  (bar chart per dataset × model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overfit_gap(lstm_runs, qnlp_runs, out_dir,
                     train_sizes=None, datasets=None):
    if train_sizes is None:
        train_sizes = [10, 50, 100]
    if datasets is None:
        datasets = ['bach', 'sst2', 'synthetic']

    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        x      = np.arange(len(train_sizes))
        width  = 0.35
        l_gaps = []
        q_gaps = []
        for n in train_sizes:
            lm = [r for r in lstm_runs
                  if r['cfg'].get('data_source') == ds
                  and int(r['cfg'].get('num_train', 0)) == n]
            qm = [r for r in qnlp_runs
                  if r['cfg'].get('data_source') == ds
                  and int(r['cfg'].get('num_train', 0)) == n]
            l_gaps.append(overfit_gap(lm) if lm else 0.0)
            q_gaps.append(overfit_gap(qm) if qm else 0.0)

        bars1 = ax.bar(x - width/2, l_gaps, width, color=LSTM_C,
                       alpha=0.85, label='LSTM')
        bars2 = ax.bar(x + width/2, q_gaps, width, color=QNLP_C,
                       alpha=0.85, label='QNLP')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'N={n}' for n in train_sizes])
        ax.set_ylabel('Overfit gap (val loss − train loss)')
        ax.set_title(f'{ds.upper()}', fontweight='bold')
        ax.legend(fontsize=9)

        # Colour bars: positive = red tint, negative = blue tint
        for bar in bars1:
            bar.set_edgecolor('white')
        for bar in bars2:
            bar.set_edgecolor('white')

    fig.suptitle('Overfit Gap by Dataset and Training Size\n'
                 '(Negative = val generalizes better than train)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'overfit_gap_comparison.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Variance (std dev) comparison at N=10 across datasets
#           Highlights QNLP stability advantage
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_n10(lstm_runs, qnlp_runs, out_dir,
                      datasets=None, n_train=10):
    if datasets is None:
        datasets = ['bach', 'sst2', 'synthetic']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x      = np.arange(len(datasets))
    width  = 0.35
    l_stds = []
    q_stds = []

    for ds in datasets:
        lm = [r for r in lstm_runs
              if r['cfg'].get('data_source') == ds
              and int(r['cfg'].get('num_train', 0)) == n_train]
        qm = [r for r in qnlp_runs
              if r['cfg'].get('data_source') == ds
              and int(r['cfg'].get('num_train', 0)) == n_train]
        _, ls = aggregate(lm, metric='val_f1', use_best=True)
        _, qs = aggregate(qm, metric='val_f1', use_best=True)
        l_stds.append(ls)
        q_stds.append(qs)

    ax.bar(x - width/2, l_stds, width, color=LSTM_C, alpha=0.85, label='LSTM')
    ax.bar(x + width/2, q_stds, width, color=QNLP_C, alpha=0.85, label='QNLP')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets])
    ax.set_ylabel('Std dev of best val F1 across seeds')
    ax.set_title(f'Model Instability at $N_{{train}}={n_train}$\n'
                 '(Lower = more stable across random seeds)',
                 fontweight='bold')
    ax.legend(fontsize=10)

    # Annotate the synthetic case
    for i, (ls, qs) in enumerate(zip(l_stds, q_stds)):
        if ls > 0.1:   # only annotate interesting bars
            ax.text(i - width/2, ls + 0.005, f'{ls:.3f}',
                    ha='center', va='bottom', fontsize=8, color=LSTM_C)
        if qs > 0:
            ax.text(i + width/2, qs + 0.005, f'{qs:.3f}',
                    ha='center', va='bottom', fontsize=8, color=QNLP_C)

    plt.tight_layout()
    savefig(fig, f'variance_comparison_n{n_train}.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Generate all report figures')
    p.add_argument('--lstm_dir', default=os.path.join(PROJECT_ROOT, 'output_classical'),
                   help='Root directory containing LSTM run subdirs')
    p.add_argument('--qnlp_dir', default=os.path.join(PROJECT_ROOT, 'output_qnlp'),
                   help='Root directory containing QNLP run subdirs')
    p.add_argument('--out_dir',  default=os.path.join(PROJECT_ROOT, 'report_figures'),
                   help='Directory to save all output figures')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading runs...')
    lstm_runs = load_runs(args.lstm_dir)
    qnlp_runs = load_runs(args.qnlp_dir)

    # ── 1. Scaling analysis (uses hardcoded timing data) ─────────────────────
    print('\n[1/6] Scaling analysis...')
    plot_scaling(args.out_dir)

    # ── 2. Training curves — Bach N=50 (most impactful for report) ───────────
    print('\n[2/6] Training curves — Bach N=50...')
    # Try all three seeds, generate one figure per seed found
    for seed in [42, 100, 2026]:
        plot_training_curves(
            lstm_runs, qnlp_runs,
            dataset='bach', n_train=50, seed=seed,
            out_dir=args.out_dir,
            filename=f'loss_bach_n50_seed{seed}.png'
        )

    # ── 3. Training curves — Synthetic N=10 (variance story) ─────────────────
    print('\n[3/6] Training curves — Synthetic N=10...')
    for seed in [42, 100, 2026]:
        plot_training_curves(
            lstm_runs, qnlp_runs,
            dataset='synthetic', n_train=10, seed=seed,
            out_dir=args.out_dir,
            filename=f'curves_synthetic_n10_seed{seed}.png'
        )

    # ── 4. F1 vs training size — all datasets ────────────────────────────────
    print('\n[4/6] F1 vs training size...')
    for ds in ['bach', 'sst2', 'synthetic']:
        # Bach QNLP only goes to N=100; pass appropriate sizes
        if ds == 'bach':
            lstm_sizes = [10, 50, 100, 250, 500]
            qnlp_sizes = [10, 50, 100]
            # plot separately so legend is correct
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for runs, color, label, sizes, marker in [
                (lstm_runs, LSTM_C, 'LSTM', lstm_sizes, 'D'),
                (qnlp_runs, QNLP_C, 'QNLP', qnlp_sizes, 'o'),
            ]:
                means, stds = [], []
                xs = []
                for n in sizes:
                    matching = [r for r in runs
                                if r['cfg'].get('data_source') == ds
                                and int(r['cfg'].get('num_train', 0)) == n]
                    if matching:
                        m, s = aggregate(matching, metric='val_f1', use_best=True)
                        means.append(m); stds.append(s); xs.append(n)
                if xs:
                    ax.errorbar(xs, means, yerr=stds, color=color,
                                marker=marker, lw=2, ms=7, capsize=4,
                                label=label)
            ax.set_xlabel('Training size $N_{train}$')
            ax.set_ylabel('Best validation F1 (mean ± std)')
            ax.set_title('Validation F1 vs Training Size — BACH', fontweight='bold')
            ax.legend(fontsize=10)
            ax.set_xscale('log')
            plt.tight_layout()
            savefig(fig, 'f1_vs_trainsize_bach.png', args.out_dir)
        else:
            plot_f1_vs_trainsize(lstm_runs, qnlp_runs, ds, args.out_dir)

    # ── 5. Overfit gap bar chart ──────────────────────────────────────────────
    print('\n[5/6] Overfit gap comparison...')
    plot_overfit_gap(lstm_runs, qnlp_runs, args.out_dir,
                     train_sizes=[10, 50, 100],
                     datasets=['bach', 'sst2', 'synthetic'])

    # ── 6. Variance at N=10 ───────────────────────────────────────────────────
    print('\n[6/6] Variance comparison at N=10...')
    plot_variance_n10(lstm_runs, qnlp_runs, args.out_dir,
                      datasets=['bach', 'sst2', 'synthetic'],
                      n_train=10)

    print(f'\n✓ All figures saved to: {args.out_dir}/')
    print('Upload ALL .png files from that folder to your Overleaf project.')
    print('\nFiles generated:')
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(args.out_dir, f)) // 1024
            print(f'  {f}  ({size_kb} KB)')


if __name__ == '__main__':
    main()