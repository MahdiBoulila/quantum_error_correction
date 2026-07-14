"""Monte Carlo experiment driver for the repetition-code circuits.

Each trial resolves its random Pauli errors into concrete gates *before*
simulation (via the noise_fn hook in circuits.build_repetition_code), so a
single shots=1 statevector simulation per trial is exact -- no shot noise.
Stabilizer measurement outcomes are deterministic given a definite injected
Pauli error, so shots=1 is not an approximation here, it's exact.

Circuits are batched into a single AerSimulator.run() call for speed.
"""
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .circuits import build_repetition_code
from .noise import sample_error

_SIM = AerSimulator(method="statevector")

_ZERO = np.array([1, 0], dtype=complex)
_ONE = np.array([0, 1], dtype=complex)


def logical_input_state(theta, phi):
    ref = Statevector.from_label("0")
    ref = ref.evolve(_ry_rz_circuit(theta, phi))
    return ref.data


def _ry_rz_circuit(theta, phi):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


def target_statevector(psi, a0, a1):
    """Ideal post-decode state: psi on d0, |00> on d1,d2, measured (a0,a1) on ancillas."""
    a1s = _ONE if a1 else _ZERO
    a0s = _ONE if a0 else _ZERO
    t = np.kron(a1s, a0s)
    t = np.kron(t, _ZERO)  # d2
    t = np.kron(t, _ZERO)  # d1
    t = np.kron(t, psi)    # d0
    return t


def random_angles(rng):
    """Haar-uniform single-qubit state, expressed as Ry/Rz angles."""
    u1, u2 = rng.random(), rng.random()
    theta = np.arccos(1 - 2 * u1)
    phi = 2 * np.pi * u2
    return theta, phi


def _qubit_label(q):
    return f"{q._register.name}{q._index}"


def idle_noise_fn(p, kind, rng, log):
    """Errors only at the post-encode idle slot -- textbook environmental noise."""
    def fn(site, qubits):
        if site != "idle_post_encode":
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
    """Errors at every gate site (rate p_gate) plus the idle slot (rate p_idle)."""
    def fn(site, qubits):
        p = p_idle if site == "idle_post_encode" else p_gate
        errs = {}
        for q in qubits:
            e = sample_error(p, "depolarizing", rng)
            if e != "I":
                errs[q] = e
                log.append({"site": site, "qubit": _qubit_label(q), "pauli": e})
        return errs
    return fn


def _parse_counts_key(key):
    # Qiskit bitstring: rightmost char = classical bit 0 (a0), next = bit 1 (a1)
    a0 = int(key[-1])
    a1 = int(key[-2])
    return a0, a1


def run_batch(basis, noise_fn_maker, n_trials, rng, correction=True, fixed_angles=None):
    """Build n_trials circuits, simulate them in one Aer call, return a DataFrame.

    noise_fn_maker(rng, log) -> noise_fn, called once per trial (fresh log list).
    fixed_angles: optional (theta, phi) to use for every trial instead of random.
    """
    circuits, logs, angles = [], [], []
    for _ in range(n_trials):
        theta, phi = fixed_angles if fixed_angles else random_angles(rng)
        log = []
        noise_fn = noise_fn_maker(rng, log)
        qc, data, anc, syn = build_repetition_code(basis, theta, phi, noise_fn=noise_fn,
                                                     correction=correction)
        circuits.append(qc)
        logs.append(log)
        angles.append((theta, phi))

    result = _SIM.run(circuits, shots=1).result()

    rows = []
    for i, qc in enumerate(circuits):
        theta, phi = angles[i]
        psi = logical_input_state(theta, phi)
        counts = result.get_counts(i)
        key = list(counts.keys())[0]
        a0, a1 = _parse_counts_key(key)
        sv = result.get_statevector(i)
        target = target_statevector(psi, a0, a1)
        fid = float(np.abs(np.vdot(target, sv.data)) ** 2)
        sites_hit = ",".join(sorted({e["site"] for e in logs[i]})) if logs[i] else ""
        paulis_hit = ",".join(sorted({e["pauli"] for e in logs[i]})) if logs[i] else ""
        rows.append({
            "trial": i, "theta": theta, "phi": phi,
            "syn_a0": a0, "syn_a1": a1,
            "n_errors": len(logs[i]), "sites_hit": sites_hit, "paulis_hit": paulis_hit,
            "fidelity": fid, "success": fid > 0.999,
        })
    df = pd.DataFrame(rows)
    df.attrs["logs"] = logs
    return df


def sweep_idle_noise(basis, p_values, n_trials, kind, seed=0, correction=True):
    """Logical error rate vs physical error probability p, for the idle noise model."""
    rng = np.random.default_rng(seed)
    records = []
    for p in p_values:
        df = run_batch(basis, lambda r, log, p=p: idle_noise_fn(p, kind, r, log),
                        n_trials, rng, correction=correction)
        records.append({
            "p": p,
            "logical_error_rate": 1 - df["success"].mean(),
            "n_trials": n_trials,
        })
    return pd.DataFrame(records)


def _uncoded_batch(kind, p, n_trials, rng):
    from qiskit import QuantumCircuit
    circuits, angles = [], []
    for _ in range(n_trials):
        theta, phi = random_angles(rng)
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.rz(phi, 0)
        e = sample_error(p, kind, rng)
        if e != "I":
            getattr(qc, e.lower())(0)
        qc.save_statevector()
        circuits.append(qc)
        angles.append((theta, phi))
    result = _SIM.run(circuits, shots=1).result()
    successes = []
    for i in range(n_trials):
        theta, phi = angles[i]
        psi = logical_input_state(theta, phi)
        sv = result.get_statevector(i)
        fid = float(np.abs(np.vdot(psi, sv.data)) ** 2)
        successes.append(fid > 0.999)
    return np.mean(successes)


def sweep_uncoded_baseline(p_values, n_trials, kind, seed=0):
    """A single physical qubit exposed directly to the same error channel, no code at all."""
    rng = np.random.default_rng(seed)
    records = []
    for p in p_values:
        success_rate = _uncoded_batch(kind, p, n_trials, rng)
        records.append({"p": p, "logical_error_rate": 1 - success_rate, "n_trials": n_trials})
    return pd.DataFrame(records)


def run_gate_level_attribution(basis, p_gate, p_idle, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    df = run_batch(basis, lambda r, log: gate_level_noise_fn(p_gate, p_idle, r, log),
                    n_trials, rng, correction=True)
    return df
