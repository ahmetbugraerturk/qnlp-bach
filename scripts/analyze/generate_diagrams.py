"""
generate_diagrams_v2.py
=======================
Generates cups_reader string diagrams + IQP circuits for
n_layers=1 AND n_layers=2, INCLUDING Qiskit circuit renders.

Output folder: example_s2c/

Run:
    python generate_diagrams_v2.py
"""

import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from lambeq import cups_reader, IQPAnsatz, AtomicType

# NEW: Qiskit conversion imports
from pytket.extensions.qiskit import tk_to_qiskit


# ── Output folder ────────────────────────────────────────────────────────────
OUT = "example_s2c/"
os.makedirs(OUT, exist_ok=True)

print(f"Output folder: {OUT}/")


# ── Sentences ────────────────────────────────────────────────────────────────
EXAMPLES = [
    # (sentence, dataset_tag, label, prefix,
    #  figsize_diag, figsize_circ_1L, figsize_circ_2L)

    (
        "good student works",
        "Synthetic", 1, "synthetic_pos",
        (7, 3.5), (8, 3.5), (12, 3.5),
    ),

    (
        "bad model fails",
        "Synthetic", 0, "synthetic_neg",
        (7, 3.5), (8, 3.5), (12, 3.5),
    ),

    (
        "entertaining movie",
        "SST-2", 1, "sst2_pos",
        (6, 3.5), (7, 3.5), (10, 3.5),
    ),

    (
        "bodily function jokes",
        "SST-2", 0, "sst2_neg",
        (7, 3.5), (9, 3.5), (13, 3.5),
    ),

    (
        "C5_quarter C5_eighth C5_16th C5_quarter C6_eighth G5_16th",
        "Bach Major", 1, "bach_major",
        (10, 3.5), (12, 3.5), (18, 3.5),
    ),

    (
        "E5_quarter E5_quarter E5_quarter E5_quarter",
        "Bach Minor", 0, "bach_minor",
        (8, 3.5), (10, 3.5), (15, 3.5),
    ),
]

def auto_figsize(circuit, base_height=4):
    n_qubits = len(circuit.cod)
    n_gates = len(circuit.boxes)

    width = max(10, n_gates * 0.7)
    height = max(base_height, n_qubits * 0.8)

    return (width, height)


# ── Helper save function ─────────────────────────────────────────────────────
def save(path):
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  saved -> {path}")


# ── Main loop ────────────────────────────────────────────────────────────────
for (
    sentence,
    dataset,
    label,
    prefix,
    fs_diag,
    fs_circ_1,
    fs_circ_2
) in EXAMPLES:

    print(f"\n[{dataset}  label={label}]  \"{sentence}\"")

    # -----------------------------------------------------------------------
    # Build DisCoCat diagram
    # -----------------------------------------------------------------------
    diagram = cups_reader.sentence2diagram(sentence)

    # -----------------------------------------------------------------------
    # 1. String diagram
    # -----------------------------------------------------------------------
    diagram.draw(
        figsize=fs_diag,
        fontsize=12,
        use_tikz=False
    )

    plt.suptitle(
        f"{dataset} | label={label}\n"
        f"cups_reader diagram | \"{sentence}\"",
        fontsize=9,
        y=1.04,
    )

    save(os.path.join(OUT, f"{prefix}_diagram.png"))

    # =======================================================================
    # 2. IQP circuit n_layers=1
    # =======================================================================
    ansatz1 = IQPAnsatz(
        {
            AtomicType.NOUN: 1,
            AtomicType.SENTENCE: 1
        },
        n_layers=1,
    )

    circuit1 = ansatz1(diagram)

    # ---- lambeq render ----
    fs_circ_1 = auto_figsize(circuit1)
    circuit1.draw(
        figsize=(16, 4),

        fontsize=8,

        asymmetry=0.15,

        boxpad=0.05,

        use_tikz=True,

        wire_labels=False,

        dom_labels={
            0: "q0",
            1: "q1",
            2: "q2",
            3: "q3",
        }
    )

    plt.suptitle(
        f"{dataset} | label={label} | n_layers=1\n"
        f"IQP circuit | \"{sentence}\"",
        fontsize=9,
        y=1.04,
    )

    save(os.path.join(OUT, f"{prefix}_circuit_1layer.png"))

    # -----------------------------------------------------------------------
    # Convert to Qiskit
    # -----------------------------------------------------------------------
    import numpy as np

    # Convert to tket
    tk_circ1 = circuit1.to_tk()

    # Get symbolic parameters
    symbols = tk_circ1.free_symbols()

    # Assign random values to parameters
    symbol_map = {
        s: np.random.uniform(0, 2*np.pi)
        for s in symbols
    }

    # Substitute parameters
    tk_circ1.symbol_substitution(symbol_map)

    # Convert to Qiskit
    qiskit_circ1 = tk_to_qiskit(tk_circ1)

    # Print textual circuit
    print("\nQiskit circuit (1 layer):")
    print(qiskit_circ1)

    # Save Qiskit render
    fig1 = qiskit_circ1.draw(
        output='mpl',
        fold=-1
    )

    fig1.savefig(
        os.path.join(OUT, f"{prefix}_qiskit_1layer.png"),
        dpi=200,
        bbox_inches='tight'
    )

    plt.close(fig1)

    print(f"  saved -> {prefix}_qiskit_1layer.png")

    # =======================================================================
    # 3. IQP circuit n_layers=2
    # =======================================================================
    ansatz2 = IQPAnsatz(
        {
            AtomicType.NOUN: 1,
            AtomicType.SENTENCE: 1
        },
        n_layers=2,
    )

    circuit2 = ansatz2(diagram)

    # ---- lambeq render ----
    fs_circ_2 = auto_figsize(circuit2)
    circuit2.draw(
        figsize=(16, 4),
        fontsize=8,
        asymmetry=0.15,
        boxpad=0.05,
        use_tikz=False,
        wire_labels=False,
        show_types=False
    )

    plt.suptitle(
        f"{dataset} | label={label} | n_layers=2\n"
        f"IQP circuit | \"{sentence}\"",
        fontsize=9,
        y=1.04,
    )

    save(os.path.join(OUT, f"{prefix}_circuit_2layer.png"))

    # -----------------------------------------------------------------------
    # Convert to Qiskit
    # -----------------------------------------------------------------------
    import numpy as np

    # Convert to tket
    tk_circ2 = circuit2.to_tk()

    # Get symbolic parameters
    symbols = tk_circ2.free_symbols()

    # Assign random values to parameters
    symbol_map = {
        s: np.random.uniform(0, 2*np.pi)
        for s in symbols
    }

    # Substitute parameters
    tk_circ2.symbol_substitution(symbol_map)

    # Convert to Qiskit
    qiskit_circ2 = tk_to_qiskit(tk_circ2)

    # Print textual circuit
    print("\nQiskit circuit (2 layers):")
    print(qiskit_circ2)

    # Save Qiskit render
    fig2 = qiskit_circ2.draw(
        output='mpl',
        fold=-1
    )

    fig2.savefig(
        os.path.join(OUT, f"{prefix}_qiskit_2layer.png"),
        dpi=200,
        bbox_inches='tight'
    )

    plt.close(fig2)

    print(f"  saved -> {prefix}_qiskit_2layer.png")


# ── Done ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Done. All files saved to: {OUT}/")
print(f"{'='*60}")