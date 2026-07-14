"""
Distance-3 rotated surface code ("Surface-17": 9 data qubits + 8 ancilla).

Data qubit layout (index = row*3+col):
    0 1 2
    3 4 5
    6 7 8

Stabilizers (verified algebraically -- commute, 8 independent generators,
logical X/Z anticommute, code distance 3):
    X-stabilizers: {0,1,3,4}, {4,5,7,8}, {2,5}, {3,6}
    Z-stabilizers: {1,2,4,5}, {3,4,6,7}, {0,1}, {7,8}
    logical X = X0 X1 X2 (top row), logical Z = Z0 Z3 Z6 (left column)

Because X-stabilizers commute with logical Z and Z-stabilizers commute with
logical X, a single circuit only ever needs to measure ONE stabilizer type:
    - readout='Z': prepare |0>^9 (a trivial +1 Z-stabilizer eigenstate),
      inject noise, measure the 4 Z-stabilizers (this is exactly the
      syndrome for X-type/bit-flip errors), correct with X gates, then
      directly read out logical Z. Never needs to touch X-stabilizers,
      so there's no "random initial syndrome" problem to work around.
    - readout='X': the mirror image, starting from |+>^9.
This is also how real surface-code experiments report results: a bit-flip
logical error rate (Z-basis prep/readout) and a phase-flip logical error
rate (X-basis prep/readout), measured as two separate ensembles.
"""
from itertools import combinations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

DATA_POS = {0: (0, 0), 1: (0, 1), 2: (0, 2),
            3: (1, 0), 4: (1, 1), 5: (1, 2),
            6: (2, 0), 7: (2, 1), 8: (2, 2)}

X_STAB = [{0, 1, 3, 4}, {4, 5, 7, 8}, {2, 5}, {3, 6}]
Z_STAB = [{1, 2, 4, 5}, {3, 4, 6, 7}, {0, 1}, {7, 8}]
LOGICAL_X = {0, 1, 2}
LOGICAL_Z = {0, 3, 6}

_PAULI_GATE = {"X": "x", "Y": "y", "Z": "z"}


def _build_lookup(stabilizers):
    """syndrome tuple -> a qubit to correct.

    Two data qubits can share a syndrome when their XOR is itself a
    stabilizer (this happens at weight-2 boundary stabilizers, e.g. {2,5}
    is a real X-stabilizer, so X-errors on qubit 2 and qubit 5 are
    genuinely degenerate: correcting either one leaves the same net effect
    on the code space, differing only by that stabilizer). Any one of the
    degenerate qubits is an equally valid correction target.
    """
    table = {}
    for q in range(9):
        syn = tuple(1 if q in s else 0 for s in stabilizers)
        table.setdefault(syn, q)
    return table


X_ERROR_LOOKUP = _build_lookup(Z_STAB)   # Z-stabilizer syndrome -> which qubit had an X error
Z_ERROR_LOOKUP = _build_lookup(X_STAB)   # X-stabilizer syndrome -> which qubit had a Z error


def _syndrome_value(syn_tuple):
    """Pack a 4-bit syndrome tuple into the classical-register integer Qiskit expects
    (bit i of the register == syn_tuple[i])."""
    return sum(b << i for i, b in enumerate(syn_tuple))


def sites_for(readout):
    stabs = Z_STAB if readout == "Z" else X_STAB
    sites = ["prep"] + [f"stab{i}_cx{j}" for i, s in enumerate(stabs) for j in range(len(s))]
    sites.append("idle_post_prep")
    sites.append("logical_readout")
    return sites


def build_surface_code_trial(readout, noise_fn=None, correction=True):
    """Build one round of: prepare logical basis state -> noise -> syndrome
    extraction (single stabilizer type) -> correction -> logical readout.

    readout: 'Z' (bit-flip protection, logical |0>_L) or 'X' (phase-flip
        protection, logical |+>_L).
    noise_fn(site_label, qubits) -> {qubit: pauli}, called at every site.
    correction: if False, syndrome is still measured but no fix applied.
    """
    if readout not in ("Z", "X"):
        raise ValueError("readout must be 'Z' or 'X'")

    data = QuantumRegister(9, "d")
    stab_anc = QuantumRegister(4, "s")
    log_anc = QuantumRegister(1, "l")
    syn = ClassicalRegister(4, "syn")
    lout = ClassicalRegister(1, "lout")
    qc = QuantumCircuit(data, stab_anc, log_anc, syn, lout)

    def noise(site, qubits):
        if noise_fn is None:
            return
        errs = noise_fn(site, qubits) or {}
        for q, lbl in errs.items():
            getattr(qc, _PAULI_GATE[lbl].lower())(q)

    stabilizers = Z_STAB if readout == "Z" else X_STAB
    lookup = X_ERROR_LOOKUP if readout == "Z" else Z_ERROR_LOOKUP
    correction_gate = "x" if readout == "Z" else "z"

    # --- prepare logical basis state ---
    if readout == "X":
        for q in range(9):
            qc.h(data[q])
    noise("prep", list(data))

    # --- noise acts on the "at rest" encoded qubits, before any further gates ---
    noise("idle_post_prep", list(data))

    # --- syndrome extraction (single stabilizer type) ---
    for i, support in enumerate(stabilizers):
        qubits = sorted(support)
        if readout == "Z":
            # Z-stabilizer: ancilla starts |0>, CX(data -> ancilla) for each data qubit
            for j, q in enumerate(qubits):
                qc.cx(data[q], stab_anc[i])
                noise(f"stab{i}_cx{j}", [data[q], stab_anc[i]])
        else:
            # X-stabilizer: ancilla in |+>, CX(ancilla -> data) for each data qubit
            qc.h(stab_anc[i])
            for j, q in enumerate(qubits):
                qc.cx(stab_anc[i], data[q])
                noise(f"stab{i}_cx{j}", [data[q], stab_anc[i]])
            qc.h(stab_anc[i])
        qc.measure(stab_anc[i], syn[i])

    # --- classically-conditioned correction via the single-error lookup table ---
    if correction:
        for syn_tuple, q in lookup.items():
            if all(b == 0 for b in syn_tuple):
                continue
            with qc.if_test((syn, _syndrome_value(syn_tuple))):
                getattr(qc, correction_gate)(data[q])

    # --- logical readout: parity of the logical operator's support qubits ---
    support = sorted(LOGICAL_Z if readout == "Z" else LOGICAL_X)
    if readout == "X":
        for q in support:
            qc.h(data[q])
    for q in support:
        qc.cx(data[q], log_anc[0])
    noise("logical_readout", [data[q] for q in support] + [log_anc[0]])
    qc.measure(log_anc[0], lout[0])
    if readout == "X":
        for q in support:
            qc.h(data[q])

    return qc, data, stab_anc, log_anc, syn, lout
