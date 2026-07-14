"""
Example 2: 3-qubit phase-flip repetition code.

Same idea as example 1, but the code is built to protect against Z (phase
flip) errors instead of X (bit flip) errors -- it's the same circuit
conjugated by Hadamards. Confirms the "dual" code behaves identically to the
bit-flip code under its own conjugate error channel.

Run: python examples/02_phase_flip_basic.py   (using the qiskit-env interpreter)
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
    qc_template, *_ = build_repetition_code("phaseflip", 0.9, 1.3, noise_fn=None, correction=True)
    print_circuit(qc_template, "Phase-flip code circuit (no noise injected here -- structure only)")

    p_values = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4]
    n_trials = 4000

    print("\nRunning phase-flip code sweep (corrected)...")
    df_corrected = sweep_idle_noise("phaseflip", p_values, n_trials, kind="Z", seed=1, correction=True)
    print(df_corrected)

    print("Running phase-flip code sweep (syndrome measured, correction skipped)...")
    df_uncorrected = sweep_idle_noise("phaseflip", p_values, n_trials, kind="Z", seed=1, correction=False)

    print("Running uncoded baseline...")
    df_uncoded = sweep_uncoded_baseline(p_values, n_trials, kind="Z", seed=1)

    out = os.path.join(RESULTS, "02_phaseflip_logical_vs_physical.png")
    plot_logical_vs_physical(df_corrected, df_uncoded,
                              "Phase-flip code: logical vs physical error rate", out,
                              df_uncorrected=df_uncorrected)
    print("saved", out)

    p_detail = 0.1
    rng = np.random.default_rng(42)
    df_detail = run_batch("phaseflip", lambda r, log: idle_noise_fn(p_detail, "Z", r, log),
                           n_trials=3000, rng=rng, correction=True)
    print(f"\nAt p={p_detail}: logical error rate = {1 - df_detail['success'].mean():.4f}")

    out = os.path.join(RESULTS, "02_phaseflip_syndrome_histogram.png")
    plot_syndrome_histogram(df_detail, f"Phase-flip code syndromes at p={p_detail}", out)
    print("saved", out)

    out = os.path.join(RESULTS, "02_phaseflip_error_count_vs_outcome.png")
    plot_error_count_vs_outcome(df_detail, f"Phase-flip code: injected error count at p={p_detail}", out)
    print("saved", out)
