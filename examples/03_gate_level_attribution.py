"""
Example 3: gate-level noise attribution.

Instead of one lump "idle noise" channel, every gate in the circuit (each CX,
each H) gets its own chance of a depolarizing (X/Y/Z) error, plus the same
idle slot from examples 1/2 representing background decoherence not tied to
any gate. For every failed trial we record which site(s) had an active
error, so we can answer: when the code fails, where did the fatal error
actually happen -- a specific CX, a basis-change H, or gate-uncorrelated
idle noise?

Run: python examples/03_gate_level_attribution.py   (using the qiskit-env interpreter)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from qec.experiment import run_gate_level_attribution, gate_level_noise_fn
from qec.circuits import build_repetition_code
from qec.display import print_circuit
from qec.viz import plot_site_attribution, plot_pauli_attribution, plot_error_count_vs_outcome

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

if __name__ == "__main__":
    n_trials = 6000
    p_gate = 0.02   # error probability per gate operation
    p_idle = 0.02   # error probability at the idle slot (background decoherence)

    for basis, label in [("bitflip", "bitflip"), ("phaseflip", "phaseflip")]:
        # one concrete sampled trial, with its actual injected error gates included,
        # so you can see what a noisy trial circuit really looks like (not just the template)
        rng = np.random.default_rng(99)
        log = []
        qc_sample, *_ = build_repetition_code(
            basis, 0.9, 1.3, noise_fn=gate_level_noise_fn(p_gate, p_idle, rng, log), correction=True)
        print_circuit(qc_sample, f"{label} code, one sampled noisy trial "
                                  f"(p_gate={p_gate}, p_idle={p_idle})")
        print("injected errors in this sample:", log if log else "(none this time)")

        print(f"\n=== {label} code, gate-level noise (p_gate={p_gate}, p_idle={p_idle}) ===")
        df = run_gate_level_attribution(basis, p_gate, p_idle, n_trials, seed=7)
        logical_error_rate = 1 - df["success"].mean()
        print(f"trials: {n_trials}, logical error rate: {logical_error_rate:.4f}")
        print(f"failed trials: {(~df.success).sum()}")

        out = os.path.join(RESULTS, f"03_{label}_site_attribution.png")
        plot_site_attribution(df, f"{label} code: where failures came from (circuit site)", out)
        print("saved", out)

        out = os.path.join(RESULTS, f"03_{label}_pauli_attribution.png")
        plot_pauli_attribution(df, f"{label} code: Pauli type present in failed trials", out)
        print("saved", out)

        out = os.path.join(RESULTS, f"03_{label}_error_count_vs_outcome.png")
        plot_error_count_vs_outcome(df, f"{label} code: injected error count per trial", out)
        print("saved", out)
