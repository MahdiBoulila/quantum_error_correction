"""
Example 4: distance-3 rotated surface code (9 data + 8 ancilla qubits).

Two separate ensembles, mirroring examples 1/2 but for the surface code:
  - 'Z' readout: logical |0>_L, independent X errors at an idle slot, the
    4 Z-stabilizers are measured and used to correct bit-flip errors.
  - 'X' readout: logical |+>_L, independent Z errors, the 4 X-stabilizers
    correct phase-flip errors.
Both should show the same qualitative benchmark as the repetition code
(quadratic suppression of logical error rate) but the surface code can
correct any single-qubit error anywhere on a 2D patch, not just 3 qubits
in a line.

Run: python examples/04_surface_code_basic.py   (using the qiskit-env interpreter)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qec.surface_experiment import sweep_idle_noise, sweep_uncoded_baseline, run_batch, idle_noise_fn
from qec.surface_code import DATA_POS, build_surface_code_trial
from qec.display import print_circuit
from qec.viz import plot_logical_vs_physical, plot_spatial_error_heatmap, plot_error_count_vs_outcome

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

if __name__ == "__main__":
    qc_z, *_ = build_surface_code_trial("Z", noise_fn=None, correction=True)
    print_circuit(qc_z, "Surface code circuit, Z-readout / bit-flip protection (no noise injected here)")
    qc_x, *_ = build_surface_code_trial("X", noise_fn=None, correction=True)
    print_circuit(qc_x, "Surface code circuit, X-readout / phase-flip protection (no noise injected here)")

    p_values = [0.005, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
    n_trials = 2500

    print("\nRunning surface code Z-readout sweep (bit-flip protection)...")
    df_z = sweep_idle_noise("Z", p_values, n_trials, kind="X", seed=1, correction=True)
    print(df_z)

    print("Running uncoded baseline (X channel)...")
    df_uncoded_x = sweep_uncoded_baseline(p_values, n_trials, kind="X", seed=1)

    out = os.path.join(RESULTS, "04_surface_Z_logical_vs_physical.png")
    plot_logical_vs_physical(df_z, df_uncoded_x,
                              "Surface-17 (d=3): logical vs physical error rate, bit-flip channel",
                              out, code_label="surface code, corrected", theory_prefactor=None)
    print("saved", out)

    print("\nRunning surface code X-readout sweep (phase-flip protection)...")
    df_x = sweep_idle_noise("X", p_values, n_trials, kind="Z", seed=1, correction=True)
    print(df_x)

    print("Running uncoded baseline (Z channel)...")
    df_uncoded_z = sweep_uncoded_baseline(p_values, n_trials, kind="Z", seed=1)

    out = os.path.join(RESULTS, "04_surface_X_logical_vs_physical.png")
    plot_logical_vs_physical(df_x, df_uncoded_z,
                              "Surface-17 (d=3): logical vs physical error rate, phase-flip channel",
                              out, code_label="surface code, corrected", theory_prefactor=None)
    print("saved", out)

    # Detailed look at one operating point: where do the (few) failures happen on the grid?
    p_detail = 0.15
    df_detail = run_batch("Z", lambda r, log, p=p_detail: idle_noise_fn(p, "X", r, log),
                           n_trials=4000, rng=__import__("numpy").random.default_rng(42),
                           correction=True)
    print(f"\nAt p={p_detail}: logical error rate = {1 - df_detail['success'].mean():.4f}")

    out = os.path.join(RESULTS, "04_surface_spatial_heatmap.png")
    plot_spatial_error_heatmap(df_detail, f"Surface code: qubit locations of injected errors in failed trials (p={p_detail})",
                                out, DATA_POS)
    print("saved", out)

    out = os.path.join(RESULTS, "04_surface_error_count_vs_outcome.png")
    plot_error_count_vs_outcome(df_detail, f"Surface code: injected error count at p={p_detail}", out)
    print("saved", out)
