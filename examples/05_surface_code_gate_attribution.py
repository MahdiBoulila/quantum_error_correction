"""
Example 5: surface code gate-level noise attribution.

Every CX in the stabilizer-measurement circuit gets its own depolarizing
error chance, plus the idle slot for background decoherence. For failed
trials we tally which circuit site(s) had an active error (as in example 3)
AND where on the physical qubit grid those errors landed (as in example 4),
answering "where did the error happen" both structurally (which gate) and
spatially (which physical qubit).

Run: python examples/05_surface_code_gate_attribution.py   (using the qiskit-env interpreter)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from qec.surface_experiment import run_gate_level_attribution, gate_level_noise_fn
from qec.surface_code import DATA_POS, build_surface_code_trial
from qec.display import print_circuit
from qec.viz import (plot_site_attribution, plot_pauli_attribution,
                      plot_spatial_error_heatmap, plot_error_count_vs_outcome)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

if __name__ == "__main__":
    n_trials = 4000
    p_gate = 0.01
    p_idle = 0.01

    for readout, label in [("Z", "bitflip"), ("X", "phaseflip")]:
        # one concrete sampled trial, with its actual injected error gates included
        rng = np.random.default_rng(99)
        log = []
        qc_sample, *_ = build_surface_code_trial(
            readout, noise_fn=gate_level_noise_fn(p_gate, p_idle, rng, log), correction=True)
        print_circuit(qc_sample, f"Surface code ({label}), one sampled noisy trial "
                                  f"(p_gate={p_gate}, p_idle={p_idle})")
        print("injected errors in this sample:", log if log else "(none this time)")

        print(f"\n=== surface code, {label} protection ({readout}-readout), "
              f"p_gate={p_gate}, p_idle={p_idle} ===")
        df = run_gate_level_attribution(readout, p_gate, p_idle, n_trials, seed=7)
        logical_error_rate = 1 - df["success"].mean()
        print(f"trials: {n_trials}, logical error rate: {logical_error_rate:.4f}, "
              f"failed trials: {(~df.success).sum()}")

        out = os.path.join(RESULTS, f"05_surface_{label}_site_attribution.png")
        plot_site_attribution(df, f"Surface code ({label}): failures by circuit site", out)
        print("saved", out)

        out = os.path.join(RESULTS, f"05_surface_{label}_pauli_attribution.png")
        plot_pauli_attribution(df, f"Surface code ({label}): Pauli type in failed trials", out)
        print("saved", out)

        out = os.path.join(RESULTS, f"05_surface_{label}_spatial_heatmap.png")
        plot_spatial_error_heatmap(df, f"Surface code ({label}): spatial location of errors in failed trials",
                                    out, DATA_POS)
        print("saved", out)

        out = os.path.join(RESULTS, f"05_surface_{label}_error_count_vs_outcome.png")
        plot_error_count_vs_outcome(df, f"Surface code ({label}): injected error count per trial", out)
        print("saved", out)
