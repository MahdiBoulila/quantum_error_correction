"""Monte Carlo experiment driver for the distance-3 surface code.

Mirrors qec/experiment.py's design: all randomness is resolved into
concrete gates before simulation, so shots=1 statevector runs are exact,
and circuits are batched into one AerSimulator.run() call.

Each readout ensemble ('Z' or 'X') is a self-contained experiment -- see
qec/surface_code.py's docstring for why that split avoids needing to
canonicalize a random initial syndrome.
"""
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from .surface_code import build_surface_code_trial
from .noise import sample_error
from .experiment import sweep_uncoded_baseline  # noqa: F401 (re-exported for convenience)

_SIM = AerSimulator(method="statevector")


def _qubit_label(q):
    return f"{q._register.name}{q._index}"


def _parse_result_key(key):
    """Two classical registers (syn: 4 bits, lout: 1 bit) space-separated;
    identify by length rather than position since Aer's group order can vary."""
    parts = key.split(" ")
    lout_str = next(p for p in parts if len(p) == 1)
    syn_str = next(p for p in parts if len(p) == 4)
    syn = tuple(int(b) for b in reversed(syn_str))  # syn[0] = register bit 0
    return int(lout_str), syn


def idle_noise_fn(p, kind, rng, log):
    """Errors only at the post-prep idle slot -- textbook environmental noise."""
    def fn(site, qubits):
        if site != "idle_post_prep":
            return None
        errs = {}
        for q in qubits:
            e = sample_error(p, kind, rng)
            if e != "I":
                errs[q] = e
                log.append({"site": site, "qubit": _qubit_label(q), "pauli": e})
        return errs
    return fn


def gate_level_noise_fn(p_gate, p_idle, rng, log):
    """Errors at every syndrome-extraction CX (rate p_gate) plus the idle slot (p_idle)."""
    def fn(site, qubits):
        if site.startswith("stab"):
            p = p_gate
        elif site == "idle_post_prep":
            p = p_idle
        else:
            return None
        errs = {}
        for q in qubits:
            e = sample_error(p, "depolarizing", rng)
            if e != "I":
                errs[q] = e
                log.append({"site": site, "qubit": _qubit_label(q), "pauli": e})
        return errs
    return fn


def run_batch(readout, noise_fn_maker, n_trials, rng, correction=True):
    circuits, logs = [], []
    for _ in range(n_trials):
        log = []
        noise_fn = noise_fn_maker(rng, log)
        qc, *_ = build_surface_code_trial(readout, noise_fn=noise_fn, correction=correction)
        circuits.append(qc)
        logs.append(log)

    result = _SIM.run(circuits, shots=1).result()

    rows = []
    for i in range(n_trials):
        counts = result.get_counts(i)
        key = list(counts.keys())[0]
        lout, syn = _parse_result_key(key)
        sites_hit = ",".join(sorted({e["site"] for e in logs[i]})) if logs[i] else ""
        paulis_hit = ",".join(sorted({e["pauli"] for e in logs[i]})) if logs[i] else ""
        rows.append({
            "trial": i, "readout": readout, "syn": syn, "lout": lout,
            "n_errors": len(logs[i]), "sites_hit": sites_hit, "paulis_hit": paulis_hit,
            "success": lout == 0,
        })
    df = pd.DataFrame(rows)
    df.attrs["logs"] = logs
    return df


def sweep_idle_noise(readout, p_values, n_trials, kind, seed=0, correction=True):
    rng = np.random.default_rng(seed)
    records = []
    for p in p_values:
        df = run_batch(readout, lambda r, log, p=p: idle_noise_fn(p, kind, r, log),
                        n_trials, rng, correction=correction)
        records.append({"p": p, "logical_error_rate": 1 - df["success"].mean(), "n_trials": n_trials})
    return pd.DataFrame(records)


def run_gate_level_attribution(readout, p_gate, p_idle, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    df = run_batch(readout, lambda r, log: gate_level_noise_fn(p_gate, p_idle, r, log),
                    n_trials, rng, correction=True)
    return df
