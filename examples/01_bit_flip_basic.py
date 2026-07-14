"""
Example 1: 3-qubit bit-flip repetition code.

A random single-qubit state is encoded across 3 physical qubits. Independent
X (bit-flip) errors are applied to each data qubit after encoding (this is
the classic "idle/environmental noise" model -- errors are not associated
with any particular gate). We measure the syndrome, apply the correction,
decode, and check fidelity against the original input state.

Run: python examples/01_bit_flip_basic.py   (using the qiskit-env interpreter)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qec.experiment import sweep_idle_noise, sweep_uncoded_baseline, run_batch, idle_noise_fn
from qec.circuits import build_repetition_code
from qec.display import print_circuit
from qec.viz import (plot_logical_vs_physical, plot_syndrome_histogram,
                      plot_error_count_vs_outcome)
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

if __name__ == "__main__":
    qc_template, *_ = build_repetition_code("bitflip", 0.9, 1.3, noise_fn=None, correction=True)
    print_circuit(qc_template, "Bit-flip code circuit (no noise injected here -- structure only)")

    p_values = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4]
    n_trials = 4000

    print("\nRunning bit-flip code sweep (corrected)...")
    df_corrected = sweep_idle_noise("bitflip", p_values, n_trials, kind="X", seed=1, correction=True)
    print(df_corrected)

    print("Running bit-flip code sweep (syndrome measured, correction skipped)...")
    df_uncorrected = sweep_idle_noise("bitflip", p_values, n_trials, kind="X", seed=1, correction=False)

    print("Running uncoded baseline...")
    df_uncoded = sweep_uncoded_baseline(p_values, n_trials, kind="X", seed=1)

    out = os.path.join(RESULTS, "01_bitflip_logical_vs_physical.png")
    plot_logical_vs_physical(df_corrected, df_uncoded,
                              "Bit-flip code: logical vs physical error rate", out,
                              df_uncorrected=df_uncorrected)
    print("saved", out)

    # Detailed look at one operating point
    p_detail = 0.1
    rng = np.random.default_rng(42)
    df_detail = run_batch("bitflip", lambda r, log: idle_noise_fn(p_detail, "X", r, log),
                           n_trials=3000, rng=rng, correction=True)
    print(f"\nAt p={p_detail}: logical error rate = {1 - df_detail['success'].mean():.4f}")

    out = os.path.join(RESULTS, "01_bitflip_syndrome_histogram.png")
    plot_syndrome_histogram(df_detail, f"Bit-flip code syndromes at p={p_detail}", out)
    print("saved", out)

    out = os.path.join(RESULTS, "01_bitflip_error_count_vs_outcome.png")
    plot_error_count_vs_outcome(df_detail, f"Bit-flip code: injected error count at p={p_detail}", out)
    print("saved", out)
