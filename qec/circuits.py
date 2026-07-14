"""
3-qubit repetition code (bit-flip or phase-flip variant), built with Qiskit.

Layout: data register d[0:3] (d0 carries the logical qubit, encoded across
all three), ancilla register a[0:2] used for stabilizer/syndrome measurement,
classical register `syn` (2 bits) holding the measured syndrome.

The circuit uses real mid-circuit measurement + classical control flow
(`if_test`) to apply the correction, then decodes back onto d0 -- this is a
genuine syndrome-extraction-and-correction circuit, not a shortcut.

A `noise_fn(site_label, qubits) -> {qubit: pauli}` hook is invoked at every
meaningful location in the circuit (state prep, each encoding/syndrome gate,
basis-change H's, and one explicit "idle" slot between encoding and syndrome
extraction). This is what lets the same builder serve both:
  - a simple physical/idle noise model (errors only at the idle slot), and
  - a gate-level noise model (errors after every gate, for attribution).
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

_PAULI_GATE = {"X": "x", "Y": "y", "Z": "z"}

# Ordered site labels this builder can call noise_fn with. Exposed so
# experiment code can build noise models without guessing string literals.
SITES_COMMON = ["input", "cx_encode_01", "cx_encode_02", "idle_post_encode",
                 "cx_synd_0a0", "cx_synd_1a0", "cx_synd_1a1", "cx_synd_2a1"]
SITES_PHASEFLIP_EXTRA = ["h_encode", "h_to_synd_frame"]


def build_repetition_code(basis, theta, phi, noise_fn=None, correction=True):
    """Build the repetition-code circuit.

    basis: 'bitflip' (protects against X errors) or 'phaseflip' (protects
        against Z errors).
    theta, phi: state-prep angles for the logical input, Rz(phi) Ry(theta) |0>.
    noise_fn: optional callable(site_label, qubits) -> dict{qubit: pauli} or
        None, called at every site during construction.
    correction: if False, still measures the syndrome (recorded in `syn`)
        but skips the conditional correction -- useful as an "uncorrected"
        baseline that is otherwise identical to the corrected circuit.
    """
    if basis not in ("bitflip", "phaseflip"):
        raise ValueError("basis must be 'bitflip' or 'phaseflip'")

    data = QuantumRegister(3, "d")
    anc = QuantumRegister(2, "a")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(data, anc, syn)

    def noise(site, qubits):
        if noise_fn is None:
            return
        errs = noise_fn(site, qubits) or {}
        for q, lbl in errs.items():
            getattr(qc, _PAULI_GATE[lbl].lower())(q)

    qc.ry(theta, data[0])
    qc.rz(phi, data[0])
    noise("input", [data[0]])

    qc.cx(data[0], data[1])
    noise("cx_encode_01", [data[0], data[1]])
    qc.cx(data[0], data[2])
    noise("cx_encode_02", [data[0], data[2]])

    if basis == "phaseflip":
        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])
        noise("h_encode", list(data))

    # environmental / idle noise acts here, on the "at rest" encoded qubits,
    # before any further gates touch them -- not attributable to any gate.
    noise("idle_post_encode", list(data))

    if basis == "phaseflip":
        # rotate Z errors into X errors so the same CNOT syndrome circuit
        # (built for the bit-flip code) can detect them
        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])
        noise("h_to_synd_frame", list(data))

    qc.cx(data[0], anc[0])
    noise("cx_synd_0a0", [data[0], anc[0]])
    qc.cx(data[1], anc[0])
    noise("cx_synd_1a0", [data[1], anc[0]])
    qc.cx(data[1], anc[1])
    noise("cx_synd_1a1", [data[1], anc[1]])
    qc.cx(data[2], anc[1])
    noise("cx_synd_2a1", [data[2], anc[1]])

    qc.measure(anc[0], syn[0])
    qc.measure(anc[1], syn[1])

    if correction:
        with qc.if_test((syn, 0b01)):
            qc.x(data[0])
        with qc.if_test((syn, 0b11)):
            qc.x(data[1])
        with qc.if_test((syn, 0b10)):
            qc.x(data[2])

    # correction above was applied in the computational (bit-flip) frame, so
    # decoding is just the plain inverse of the encode CNOTs -- no further
    # basis change needed, we're already back in the GHZ frame.
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[1])

    qc.save_statevector()
    return qc, data, anc, syn


def sites_for(basis):
    return SITES_COMMON + (SITES_PHASEFLIP_EXTRA if basis == "phaseflip" else [])
