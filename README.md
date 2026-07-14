# Quantum Error Correction Sandbox

Small, from-scratch Qiskit simulations of quantum error correction: encode a
qubit, inject random Pauli errors, run real syndrome-extraction-and-correction
circuits (mid-circuit measurement + classical feed-forward, not a shortcut),
and visualize how much error got corrected and where it came from.

Two codes so far:
- **3-qubit repetition code** (bit-flip and phase-flip variants)
- **Distance-3 rotated surface code** ("Surface-17": 9 data + 8 ancilla qubits)

## Requirements

Everything runs against the `qiskit-env` conda environment
(`C:\Users\User\.conda\envs\qiskit-env`), not the system Python. Versions used:

- Python 3.11
- qiskit 2.3.1
- qiskit-aer 0.17.2 (statevector simulation, incl. classical control flow)
- numpy 2.4.4, pandas 2.3.3, matplotlib 3.10.8

Aer is what makes this straightforward: exact statevector simulation plus
support for `QuantumCircuit.if_test(...)` (mid-circuit measurement with
classically-conditioned gates), so the correction step is a genuine dynamic
circuit, not something faked in Python after the fact.

Run any example with the env's interpreter directly, e.g.:

```
C:\Users\User\.conda\envs\qiskit-env\python.exe examples\01_bit_flip_basic.py
```

## Folder structure

```
qec/                    reusable library code
  noise.py              samples a single Pauli error (X/Z/depolarizing) for one noise site
  circuits.py           repetition-code circuit builder (bit-flip & phase-flip)
  experiment.py          Monte Carlo driver for the repetition code
  surface_code.py        distance-3 surface code: stabilizers, lookup-table decoder, circuit builder
  surface_experiment.py  Monte Carlo driver for the surface code
  viz.py                 matplotlib plotting, shared by both codes
  display.py              prints a circuit's text diagram to the console

examples/                runnable scripts, each saves its plots to results/
results/                 generated PNGs (regenerated each time an example is run)
```

Every example prints the circuit it's about to run (via `qec/display.py`)
before running any trials, using Qiskit's text drawer. Examples 1, 2 and 4
print the clean template circuit (no noise gates, just the algorithm
structure); examples 3 and 5 additionally print one concretely-sampled noisy
trial (with the actual injected error gates included) so you can see what a
trial circuit really looks like, not just the noiseless skeleton. On Windows
this needs stdout reconfigured to UTF-8 (`display.py` does this automatically)
since the default console encoding can't render Qiskit's box-drawing characters.

The two `*_experiment.py` modules follow the same pattern: build many
concrete circuits with random errors already resolved into explicit gates
(via a `noise_fn(site_label, qubits)` hook called at every gate location
during circuit construction), batch them into a single `AerSimulator.run()`
call, then read off success/failure per trial. Because each trial's errors
are decided before simulation, a single shot per circuit is exact — no shot
noise to average out, only the Monte Carlo sampling over which errors occurred.

## Examples

**01_bit_flip_basic.py** — 3-qubit bit-flip repetition code. A random
single-qubit input is encoded across 3 qubits; independent X errors are
applied to each data qubit at an idle slot after encoding (errors from
"the environment," not tied to any gate). Plots logical vs. physical error
rate against an uncoded baseline and the 3p² theory curve, plus a syndrome
histogram and an injected-error-count histogram at one operating point.

**02_phase_flip_basic.py** — Same code conjugated by Hadamards to protect
against Z errors instead of X. Produces near-identical curves to example 1
under its own conjugate error channel, which is a correctness check on the
implementation (bit-flip and phase-flip codes should be exact duals).

**03_gate_level_attribution.py** — Instead of one lump noise channel, every
gate in the circuit (each CX, each H) gets its own depolarizing-error
chance, plus the same idle slot as examples 1/2. For every failed trial we
record which site(s) had an active error, producing bar charts of failures
by circuit location and by Pauli type (X/Y/Z). Key finding: the bit-flip
code's failures are dominated by Z/Y errors (and vice versa for phase-flip)
— it's not really "which gate" that matters, it's "which error axis,"
since a repetition code only ever protects one.

**04_surface_code_basic.py** — Distance-3 surface code version of examples
1/2. Two ensembles: 'Z' readout (logical |0>, protects against X errors)
and 'X' readout (logical |+>, protects against Z errors). Logical vs.
physical error rate crosses the bare-qubit baseline around p≈0.08–0.10,
consistent with the known surface-code threshold (~10.9%) for this noise
model — a good sanity check on the hand-derived stabilizers. Also plots a
spatial heatmap of which physical qubit (on the 3x3 grid) had an error in
failed trials.

**05_surface_code_gate_attribution.py** — Gate-level noise version of
example 4, reusing the same site/Pauli attribution plots as example 3 plus
the spatial heatmap. Key finding: errors on qubits 0, 3, 6 (the left column
of the grid, which is exactly the logical-Z operator's support) are far
more likely to cause a logical failure than errors on the opposite column
— a genuinely spatial answer to "where did the error happen," which the
1D repetition code has no equivalent of.

## Design notes worth knowing before extending this

- **Stabilizers were verified algebraically before writing any circuit
  code** (`qec/surface_code.py`'s module docstring has the derivation): all
  stabilizers commute, 8 independent generators, logical X/Z anticommute,
  brute-force-confirmed distance 3. Worth re-deriving/re-checking this way
  for any new code rather than hand-copying a stabilizer table from memory.
- The surface code experiments split into two independent single-basis
  ensembles (Z-readout / X-readout) rather than one circuit that tracks
  both simultaneously. This sidesteps a real subtlety: preparing `|0>^9`
  leaves the X-stabilizer eigenvalues randomly collapsed (they're not
  eigenstates of `|0>^9`), which would corrupt a combined-syndrome
  correction unless you first canonicalize with an extra initialization
  round. Restricting each ensemble to the one stabilizer type its readout
  actually depends on avoids needing that extra round entirely.
- The repetition-code input state is Haar-random each trial (via random
  Ry/Rz angles); the surface code instead uses fixed logical `|0>`/`|+>`
  states, since encoding an arbitrary single-qubit state into the surface
  code needs a real encoding circuit that hasn't been built yet.
- Only a single round of syndrome extraction is simulated — no repeated
  stabilizer rounds, no syndrome-difference decoding across rounds.

## Possible next steps

- Shor's 9-qubit code (concatenation of both repetition codes, corrects
  arbitrary single-qubit errors in one code rather than needing two
  separate ensembles).
- Distance-5 surface code, or multiple syndrome-extraction rounds with
  real round-to-round syndrome differencing.
- A general MWPM decoder (`networkx` is already a dependency) in place of
  the brute-force lookup table, which stops scaling once distance grows.
